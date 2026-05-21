"""
MT3 动态对准扩展 — Kalman / EKF 滤波器

实现三个核心操作：
  predict(dt)         : 时间推进（预测步）
  update(obs)         : 融合新观测（更新步）
  predict_ahead(tau)  : 延迟补偿预测（不改变内部状态）

状态向量：x = [Δx, Δy, Δθ, Δẋ, Δẏ, Δθ̇]  (6 维)
观测向量：z = [Δx, Δy, Δθ]                  (3 维)
观测矩阵：H = [I₃ | 0₃]（取前三维）

与 MotionModel 解耦
------------------
线性模型（CVModel）→ 标准 KF（F 与状态无关）
非线性模型（CTModel）→ EKF（predict_state 非线性，F_jacobian 线性化）
两者接口完全一致，由运动模型实例决定行为。

延迟补偿（设计笔记第三节）
--------------------------
predict_ahead(τ) 在当前状态基础上向前预测 τ 秒，**不修改内部状态**。
当 τ = 实际系统延迟时，控制器执行时刻的位置残差：
    Δx_residual ≈ ½ · a · τ² ≈ 0.5mm（10cm/s 场景）
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .types import ObjectObservation, TrackerState
from .motion_models import MotionModel, CVModel


# 观测矩阵 H：从 6 维状态直接取前 3 维（位置 + 角度）
_H = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]],
    dtype=float,
)


class KalmanFilter:
    """
    通用 Kalman / EKF 滤波器，追踪水平面 T_δ(t) 的 6 维状态。

    Parameters
    ----------
    model        : 运动模型，默认 CVModel()
    R_diag       : 观测噪声标准差向量 [σ_x (m), σ_y (m), σ_θ (rad)]
                   默认对应 RealSense D415 精度：4mm 位置，3° 角度
    init_vel_cov : 速度分量的初始方差 (m²/s² 或 rad²/s²)，
                   初始不知道速度时设大一些，让滤波器快速收敛
    """

    def __init__(
        self,
        model:        MotionModel | None = None,
        R_diag:       np.ndarray | None = None,
        init_vel_cov: float = 0.01,
    ) -> None:
        self.model: MotionModel = model if model is not None else CVModel()

        # 观测噪声协方差 R（3×3 对角矩阵）
        if R_diag is None:
            R_diag = np.array([0.004, 0.004, np.deg2rad(3.0)], dtype=float)
        self.R = np.diag(R_diag ** 2)

        self._init_vel_cov = float(init_vel_cov)
        self._state: TrackerState | None = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self, obs: ObjectObservation) -> None:
        """
        用第一帧观测初始化滤波器。

        位置初始化为 obs 值，速度初始化为 0（不知道初速度）。
        初始协方差：位置 ≈ 传感器噪声，速度 ≈ init_vel_cov（较大）。

        Parameters
        ----------
        obs : 初始观测（通常是 GICP 对准完成时刻的观测，此时 Δx=Δy=Δθ=0）
        """
        x0 = np.array(
            [obs.delta_x, obs.delta_y, obs.delta_theta,
             0.0,          0.0,          0.0],
            dtype=float,
        )

        P0 = np.diag([
            self.R[0, 0],               # Δx  初始方差（同观测噪声）
            self.R[1, 1],               # Δy  初始方差
            self.R[2, 2],               # Δθ  初始方差
            self._init_vel_cov,         # Δẋ  初始方差（较大，快速收敛）
            self._init_vel_cov,         # Δẏ  初始方差
            self._init_vel_cov * 0.1,   # Δθ̇ 初始方差（角速度通常较小）
        ])

        self._state = TrackerState(x=x0, P=P0, timestamp=obs.timestamp)

    # ------------------------------------------------------------------
    # 预测步（Predict）
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> TrackerState:
        """
        时间推进：用运动模型将状态分布向前传播 dt 秒。

        线性 KF:  x̂ = F·x,          P = F·P·Fᵀ + Q
        EKF:      x̂ = f(x),         P = J·P·Jᵀ + Q
                  （J = ∂f/∂x，由模型的 F_jacobian 提供）

        内部状态直接更新。

        Parameters
        ----------
        dt : 时间步长 (s)，必须 > 0

        Returns
        -------
        TrackerState : 预测后状态的副本
        """
        if self._state is None:
            raise RuntimeError("KalmanFilter 未初始化，请先调用 initialize()")
        if dt <= 0:
            raise ValueError(f"dt 必须 > 0，得到 {dt:.6f}")

        x = self._state.x
        P = self._state.P

        # 状态传播（线性模型：F@x；非线性模型：f(x)）
        x_pred = self.model.predict_state(x, dt)
        x_pred[2] = _wrap_angle(x_pred[2])   # 角度归一化

        # 协方差传播
        F_mat = self.model.F_jacobian(x, dt)
        Q     = self.model.Q_noise(dt)
        P_pred = F_mat @ P @ F_mat.T + Q

        self._state = TrackerState(
            x=x_pred,
            P=P_pred,
            timestamp=self._state.timestamp + dt,
        )
        return deepcopy(self._state)

    # ------------------------------------------------------------------
    # 更新步（Update）
    # ------------------------------------------------------------------

    def update(self, obs: ObjectObservation) -> TrackerState:
        """
        融合新观测，修正预测状态（标准 Kalman 更新方程）。

        当 obs.is_valid=False 时跳过更新，直接返回当前预测状态
        （相当于只做预测，适合点云分割失败的帧）。

        核心方程
        --------
        创新量：  y  = z − H·x̂        （角度分量归一化）
        创新协方差：S  = H·P·Hᵀ + R
        卡尔曼增益：K  = P·Hᵀ·S⁻¹
        状态更新：  x  = x̂ + K·y
        协方差更新：P  = (I−K·H)·P·(I−K·H)ᵀ + K·R·Kᵀ  （Joseph 稳定形式）

        Joseph 形式比标准 P=(I-KH)P 数值上更稳定，始终保持正定性。

        Parameters
        ----------
        obs : 当前帧观测（来自 PoseEstimator）

        Returns
        -------
        TrackerState : 更新后状态的副本
        """
        if self._state is None:
            raise RuntimeError("KalmanFilter 未初始化，请先调用 initialize()")

        # 无效观测：跳过更新，时间戳对齐
        if not obs.is_valid:
            self._state = TrackerState(
                x=self._state.x.copy(),
                P=self._state.P.copy(),
                timestamp=obs.timestamp,
            )
            return deepcopy(self._state)

        x = self._state.x
        P = self._state.P
        H = _H

        # 观测向量
        z = obs.to_array()

        # 创新量（角度分量必须归一化，避免 ±π 附近跳变）
        innovation    = z - H @ x
        innovation[2] = _wrap_angle(innovation[2])

        # 创新协方差
        S = H @ P @ H.T + self.R

        # 卡尔曼增益（用 solve 代替 inv，数值更稳定）
        K = P @ H.T @ np.linalg.solve(S.T, np.eye(3)).T

        # 状态更新
        x_new    = x + K @ innovation
        x_new[2] = _wrap_angle(x_new[2])

        # 协方差更新（Joseph 稳定形式）
        I_KH  = np.eye(6) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ self.R @ K.T

        self._state = TrackerState(
            x=x_new,
            P=P_new,
            timestamp=obs.timestamp,
        )
        return deepcopy(self._state)

    # ------------------------------------------------------------------
    # 前向预测（Predict-Ahead）— 延迟补偿核心
    # ------------------------------------------------------------------

    def predict_ahead(self, tau: float) -> TrackerState:
        """
        在当前状态基础上向前预测 tau 秒，用于补偿系统延迟。

        ⚠ 不修改内部状态——仅返回预测结果，滤波器下一次 update/predict
          仍基于当前内部状态继续工作。

        设计原理（设计笔记第三节）
        ---------------------------
        系统总延迟 τ（感知+计算+执行，约 100ms）使控制器总在
        追"过去的位置"。predict_ahead(τ) 主动抵消这一延迟：

            控制器执行时刻的位置残差：
            Δx_residual ≈ ½ · a · τ² ≈ 0.5mm（a=0.1m/s², τ=0.1s）

        Parameters
        ----------
        tau : 向前预测时长 (s)，通常等于感知+计算+执行总延迟

        Returns
        -------
        TrackerState : 预测 tau 秒后的状态，timestamp = current + tau
        """
        if self._state is None:
            raise RuntimeError("KalmanFilter 未初始化，请先调用 initialize()")

        if tau <= 0.0:
            return deepcopy(self._state)

        saved = self._state   # 不 deepcopy，只读引用

        x_ahead = self.model.predict_state(saved.x, tau)
        x_ahead[2] = _wrap_angle(x_ahead[2])

        F_mat   = self.model.F_jacobian(saved.x, tau)
        Q       = self.model.Q_noise(tau)
        P_ahead = F_mat @ saved.P @ F_mat.T + Q

        return TrackerState(
            x=x_ahead,
            P=P_ahead,
            timestamp=saved.timestamp + tau,
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def state(self) -> TrackerState | None:
        """当前滤波器状态的只读副本（未初始化时返回 None）"""
        return deepcopy(self._state) if self._state is not None else None

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        return f"KalmanFilter(model={self.model!r}, {status})"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """将任意角度归一化到 (−π, π]"""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)
