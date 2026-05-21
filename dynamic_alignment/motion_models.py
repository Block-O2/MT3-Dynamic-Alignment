"""
MT3 动态对准扩展 — 运动模型

实现 CV（匀速）和 CT（协调转弯）两种运动模型。

两种模型均以 6 维状态为统一接口：
    x = [Δx, Δy, Δθ, Δẋ, Δẏ, Δθ̇]

设计依据（设计笔记第四节）
--------------------------
对任意光滑轨迹 Taylor 展开：
    x(t+τ) = x(t) + ẋτ + ½ẍτ² + O(τ³)

低速（< 10 cm/s）、短预测窗口（τ ≈ 100ms）下，三阶项 ≈ 0.017mm，
远低于 D415 深度精度（约 3mm），因此三种模型已构成充分近似基。

CT 在 ω → 0 时自动退化为 CV（零保护阈值 ε = 1e-4 rad/s）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class MotionModel(ABC):
    """
    Kalman 预测步所需的运动模型接口。

    子类需实现三个方法：
      predict_state(x, dt)  : 状态非线性/线性传播
      F_jacobian(x, dt)     : 传播函数关于 x 的 Jacobian（或线性模型的 F）
      Q_noise(dt)           : 离散过程噪声协方差 Q
    """

    @abstractmethod
    def predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        从当前状态 x 预测 dt 秒后的状态。

        Parameters
        ----------
        x  : shape (6,)，当前状态 [Δx, Δy, Δθ, Δẋ, Δẏ, Δθ̇]
        dt : 时间步长 (s)，必须 > 0

        Returns
        -------
        x_pred : shape (6,)，预测状态
        """
        ...

    @abstractmethod
    def F_jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        预测函数 f(x) 关于 x 的 Jacobian 矩阵，shape (6, 6)。

        线性模型（CV）：F 与 x 无关，等于状态转移矩阵。
        非线性模型（CT）：EKF 线性化所需的数值 Jacobian。
        """
        ...

    @abstractmethod
    def Q_noise(self, dt: float) -> np.ndarray:
        """
        离散过程噪声协方差矩阵，shape (6, 6)。
        使用"离散白噪声加速度"（DWNA）标准形式。
        """
        ...


# ---------------------------------------------------------------------------
# CVModel — Constant Velocity（匀速）
# ---------------------------------------------------------------------------

class CVModel(MotionModel):
    """
    匀速运动模型（Constant Velocity）。

    假设速度在预测窗口内不变：
        Δx(t+dt)  = Δx  + Δẋ · dt
        Δy(t+dt)  = Δy  + Δẏ · dt
        Δθ(t+dt)  = Δθ  + Δθ̇ · dt
        Δẋ/Δẏ/Δθ̇ 保持不变（加速度作为过程噪声建模）

    这是**线性**模型，不需要 EKF；Jacobian 即转移矩阵 F，与 x 无关。

    适用场景：直线平移、速度缓慢变化，对 τ ≈ 100ms 延迟补偿已足够。

    Parameters
    ----------
    q_pos : 位置轴（x, y）的过程噪声谱密度 (m²/s³)
            默认 1e-3 对应约 3cm/s² 随机加速度扰动
    q_ang : 角度轴（θ）的过程噪声谱密度 (rad²/s³)
            默认 1e-4 对应约 0.6°/s² 扰动
    """

    def __init__(self, q_pos: float = 1e-3, q_ang: float = 1e-4) -> None:
        self.q_pos = float(q_pos)
        self.q_ang = float(q_ang)

    def predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """线性传播：x_pred = F @ x"""
        return self.F_jacobian(x, dt) @ x

    def F_jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        CV 状态转移矩阵：

            ┌                    ┐
            │ 1  0  0  dt  0   0 │
            │ 0  1  0   0  dt  0 │
            │ 0  0  1   0   0  dt│
        F = │ 0  0  0   1   0  0 │
            │ 0  0  0   0   1  0 │
            │ 0  0  0   0   0  1 │
            └                    ┘

        即 [ I  dt·I ] 的分块结构。
             [ 0    I  ]
        """
        F = np.eye(6)
        F[0, 3] = dt   # Δx  += Δẋ · dt
        F[1, 4] = dt   # Δy  += Δẏ · dt
        F[2, 5] = dt   # Δθ  += Δθ̇ · dt
        return F

    def Q_noise(self, dt: float) -> np.ndarray:
        """
        离散白噪声加速度过程噪声（DWNA 模型）。

        对每个独立轴的位置-速度对 (p, v)，噪声块为：

            Q_block = q · [ dt⁴/4   dt³/2 ]
                          [ dt³/2   dt²   ]

        这是从连续时间白噪声加速度过程离散化的标准结果。
        三个轴（x, y, θ）独立，分别使用 q_pos / q_pos / q_ang。
        """
        Q = np.zeros((6, 6))
        for i, q in enumerate([self.q_pos, self.q_pos, self.q_ang]):
            # 位置-位置块
            Q[i,   i  ] = q * dt**4 / 4.0
            # 位置-速度块（对称）
            Q[i,   i+3] = q * dt**3 / 2.0
            Q[i+3, i  ] = q * dt**3 / 2.0
            # 速度-速度块
            Q[i+3, i+3] = q * dt**2
        return Q

    def __repr__(self) -> str:
        return f"CVModel(q_pos={self.q_pos:.1e}, q_ang={self.q_ang:.1e})"


# ---------------------------------------------------------------------------
# CTModel — Coordinated Turn（协调转弯）
# ---------------------------------------------------------------------------

class CTModel(MotionModel):
    """
    协调转弯模型（Coordinated Turn）。

    适用于匀速圆弧运动（传送带弯道、弧形轨道、旋转工作台）。

    运动方程
    --------
    设当前速度幅值 v = √(Δẋ² + Δẏ²)，速度方向角 φ = atan2(Δẏ, Δẋ)，
    转率 ω = Δθ̇，转弯半径 r = v / ω（ω ≠ 0 时）：

        Δx(t+dt)  = Δx + (v/ω)·[sin(φ + ω·dt) − sin(φ)]
        Δy(t+dt)  = Δy − (v/ω)·[cos(φ + ω·dt) − cos(φ)]
        Δθ(t+dt)  = Δθ + ω·dt
        Δẋ(t+dt)  = v·cos(φ + ω·dt)     # 速度方向随转率旋转
        Δẏ(t+dt)  = v·sin(φ + ω·dt)
        Δθ̇(t+dt) = ω                    # 匀速转弯假设

    当 |ω| < ε 时自动退化为 CV，避免 v/ω 数值奇异。

    推导验证（半圈测试）
    --------------------
    从 (0,0) 以 +X 方向速度 v 出发，ω > 0（CCW），经 t = π/ω：
      x = (v/ω)·[sin(π) − 0] = 0 ✓
      y = −(v/ω)·[cos(π) − 1] = 2v/ω = 2r ✓（直径距离）

    接口
    ----
    输入/输出状态均为 [Δx, Δy, Δθ, Δẋ, Δẏ, Δθ̇]，与 CVModel 一致。
    Jacobian 使用数值微分（精度对 EKF 足够，避免手推易错的解析形式）。

    Parameters
    ----------
    q_pos     : 位置过程噪声谱密度 (m²/s³)
    q_ang     : 角度过程噪声谱密度 (rad²/s³)
    omega_eps : 转率零值保护阈值 (rad/s)，低于此值退化为 CV
    jac_eps   : 数值 Jacobian 有限差分步长
    """

    def __init__(
        self,
        q_pos:     float = 1e-3,
        q_ang:     float = 1e-4,
        omega_eps: float = 1e-4,
        jac_eps:   float = 1e-6,
    ) -> None:
        self.q_pos     = float(q_pos)
        self.q_ang     = float(q_ang)
        self.omega_eps = float(omega_eps)
        self.jac_eps   = float(jac_eps)

    def predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        非线性 CT 状态传播。

        从 (Δẋ, Δẏ) 提取速度幅值 v 和方向角 φ，
        用 CT 方程传播位移，输出格式与输入一致。
        """
        px, py, pth = x[0], x[1], x[2]
        vx, vy,  ω  = x[3], x[4], x[5]

        v   = float(np.hypot(vx, vy))    # 速度幅值（始终 ≥ 0）
        phi = float(np.arctan2(vy, vx))  # 速度方向角

        if abs(ω) < self.omega_eps:
            # ── ω ≈ 0：退化为 CV（一阶 Taylor 展开等价）──
            new_px  = px  + vx * dt
            new_py  = py  + vy * dt
            new_pth = pth + ω  * dt
            new_vx, new_vy, new_ω = vx, vy, ω

        else:
            # ── 标准 CT 圆弧积分 ──
            phi_next = phi + ω * dt      # dt 后的速度方向角

            new_px  = px  + (v / ω) * (np.sin(phi_next) - np.sin(phi))
            new_py  = py  - (v / ω) * (np.cos(phi_next) - np.cos(phi))
            new_pth = pth + ω * dt

            new_vx  = v * np.cos(phi_next)   # 速度向量随转率旋转
            new_vy  = v * np.sin(phi_next)
            new_ω   = ω                       # 匀速转弯：ω 不变

        return np.array([new_px, new_py, new_pth, new_vx, new_vy, new_ω])

    def F_jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        CT 预测函数的数值 Jacobian（前向有限差分），shape (6, 6)。

        解析 Jacobian 推导繁琐且易出错；数值差分在 jac_eps=1e-6 时
        精度约 1e-10，对 EKF 协方差传播完全足够。
        """
        x0  = self.predict_state(x, dt)
        F   = np.empty((6, 6))
        eps = self.jac_eps

        for j in range(6):
            x_pert    = x.copy()
            x_pert[j] += eps
            F[:, j]   = (self.predict_state(x_pert, dt) - x0) / eps

        return F

    def Q_noise(self, dt: float) -> np.ndarray:
        """
        与 CVModel 相同结构的 DWNA 过程噪声。
        CT 的不确定性主要来自 ω 的缓慢漂移，归入角速度噪声。
        """
        Q = np.zeros((6, 6))
        for i, q in enumerate([self.q_pos, self.q_pos, self.q_ang]):
            Q[i,   i  ] = q * dt**4 / 4.0
            Q[i,   i+3] = q * dt**3 / 2.0
            Q[i+3, i  ] = q * dt**3 / 2.0
            Q[i+3, i+3] = q * dt**2
        return Q

    def __repr__(self) -> str:
        return (
            f"CTModel(q_pos={self.q_pos:.1e}, q_ang={self.q_ang:.1e}, "
            f"omega_eps={self.omega_eps:.1e})"
        )
