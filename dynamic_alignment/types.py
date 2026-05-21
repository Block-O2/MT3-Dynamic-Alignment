"""
MT3 动态对准扩展 — 数据结构定义

所有模块共用的 dataclass，不依赖任何硬件或 ROS 库。

坐标系约定
----------
- 世界坐标系 W：右手系，Z 轴向上，单位 m / rad
- T_δ 只在水平面 (X-Y) 变化，不涉及 Z 方向位移
- 角度：绕 Z 轴右手正方向为正，归一化到 (-π, π]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# ObjectObservation  —  点云处理后的原始观测
# ---------------------------------------------------------------------------

@dataclass
class ObjectObservation:
    """
    点云估计器输出的单帧原始观测量。

    表示当前帧物体相对于 demo 参考帧的 2D 变换 [Δx, Δy, Δθ]。
    这三个量直接作为 Kalman 滤波器的观测向量 z。

    Fields
    ------
    delta_x     : 质心 X 方向位移 (m)
    delta_y     : 质心 Y 方向位移 (m)
    delta_theta : PCA 主轴旋转角 (rad)，已消除 180° 歧义
    timestamp   : 观测时间戳 (s)，用于 Kalman dt 计算
    is_valid    : False 表示本帧无效（点云太少、分割失败等），
                  Kalman 收到 is_valid=False 时跳过更新步
    """
    delta_x:     float
    delta_y:     float
    delta_theta: float
    timestamp:   float
    is_valid:    bool = True

    def to_array(self) -> np.ndarray:
        """返回观测向量 z = [Δx, Δy, Δθ]，shape (3,)"""
        return np.array([self.delta_x, self.delta_y, self.delta_theta], dtype=float)

    def __repr__(self) -> str:
        status = "OK" if self.is_valid else "INVALID"
        return (
            f"ObjectObservation("
            f"dx={self.delta_x*1e3:.1f}mm, "
            f"dy={self.delta_y*1e3:.1f}mm, "
            f"dθ={np.degrees(self.delta_theta):.1f}°, "
            f"t={self.timestamp:.3f}s, {status})"
        )


# ---------------------------------------------------------------------------
# TrackerState  —  Kalman 滤波器完整状态
# ---------------------------------------------------------------------------

@dataclass
class TrackerState:
    """
    Kalman 滤波器的均值 + 协方差，对应某一时刻的最优估计。

    状态向量 x（6 维，水平面）
    -------------------------
    x[0] = Δx    (m)      物体质心 X 方向位移
    x[1] = Δy    (m)      物体质心 Y 方向位移
    x[2] = Δθ    (rad)    物体朝向旋转角
    x[3] = Δẋ    (m/s)    X 方向速度
    x[4] = Δẏ    (m/s)    Y 方向速度
    x[5] = Δθ̇   (rad/s)  角速度

    Fields
    ------
    x         : 状态均值，shape (6,)
    P         : 协方差矩阵，shape (6, 6)，正定
    timestamp : 该状态对应的时间戳 (s)
    """
    x:         np.ndarray   # shape (6,)
    P:         np.ndarray   # shape (6, 6)
    timestamp: float

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float).reshape(6)
        self.P = np.asarray(self.P, dtype=float).reshape(6, 6)

    # ----- 便捷属性 -----

    @property
    def delta_x(self) -> float:
        return float(self.x[0])

    @property
    def delta_y(self) -> float:
        return float(self.x[1])

    @property
    def delta_theta(self) -> float:
        return float(self.x[2])

    @property
    def velocity_xy(self) -> np.ndarray:
        """平移速度向量 [Δẋ, Δẏ]，shape (2,)"""
        return self.x[3:5].copy()

    @property
    def angular_velocity(self) -> float:
        """角速度 Δθ̇ (rad/s)"""
        return float(self.x[5])

    @property
    def speed(self) -> float:
        """平移速度幅值 (m/s)"""
        return float(np.linalg.norm(self.velocity_xy))

    @property
    def position_std(self) -> np.ndarray:
        """位置估计标准差 [σ_x, σ_y]，shape (2,)，单位 m"""
        return np.sqrt(np.diag(self.P)[:2])

    def __repr__(self) -> str:
        return (
            f"TrackerState("
            f"pos=({self.delta_x*1e3:.1f}, {self.delta_y*1e3:.1f})mm, "
            f"θ={np.degrees(self.delta_theta):.1f}°, "
            f"v={self.speed*1e2:.1f}cm/s, "
            f"t={self.timestamp:.3f}s)"
        )


# ---------------------------------------------------------------------------
# DemoData  —  MT3 录制的 demo 位姿序列
# ---------------------------------------------------------------------------

@dataclass
class DemoData:
    """
    MT3 存储的末端执行器位姿序列 T_WE_demo(t)。

    MT3 论文确认每帧存 6D 位姿（世界→末端）。
    alignment 阶段只用第 0 帧；interaction 阶段按时间推进。

    Fields
    ------
    poses      : T_WE_demo(t) 列表，每个元素 shape (4, 4)，SE(3) 齐次矩阵
    timestamps : 对应时间戳列表 (s)，单调递增
    """
    poses:      List[np.ndarray]
    timestamps: List[float]

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.timestamps):
            raise ValueError(
                f"poses ({len(self.poses)}) 与 timestamps "
                f"({len(self.timestamps)}) 长度必须一致"
            )
        if len(self.poses) == 0:
            raise ValueError("DemoData 不能为空")

    @property
    def T0(self) -> np.ndarray:
        """
        Alignment 阶段的参考位姿：demo 第 0 帧 T_WE_demo(0)。
        shape (4, 4)
        """
        return self.poses[0].copy()

    @property
    def duration(self) -> float:
        """demo 总时长 (s)"""
        return float(self.timestamps[-1] - self.timestamps[0])

    def get_pose_at(self, t: float) -> np.ndarray:
        """
        线性插值获取任意时刻的 demo 位姿，shape (4, 4)。

        t 超出范围时 clamp 到端点（不外推）。

        旋转部分使用矩阵线性插值后 SVD 正交化，在短步长（<50ms）下误差
        < 0.01°，满足工程需求。如需更高精度，可替换为 scipy SLERP。
        """
        ts = np.asarray(self.timestamps, dtype=float)

        # clamp
        if t <= ts[0]:
            return self.poses[0].copy()
        if t >= ts[-1]:
            return self.poses[-1].copy()

        # 二分查找左端点
        idx = int(np.searchsorted(ts, t, side="right")) - 1
        t0, t1 = float(ts[idx]), float(ts[idx + 1])
        alpha = (t - t0) / (t1 - t0)          # ∈ [0, 1)

        T0 = self.poses[idx]
        T1 = self.poses[idx + 1]

        # 平移线性插值
        T_interp = np.eye(4)
        T_interp[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

        # 旋转：矩阵线性插值 + SVD 正交化（近似 SLERP，短步长够用）
        R_blend = (1.0 - alpha) * T0[:3, :3] + alpha * T1[:3, :3]
        U, _, Vt = np.linalg.svd(R_blend)
        T_interp[:3, :3] = U @ Vt

        return T_interp

    def __repr__(self) -> str:
        return (
            f"DemoData(frames={len(self.poses)}, "
            f"duration={self.duration:.2f}s)"
        )


# ---------------------------------------------------------------------------
# 工厂函数：从单帧位姿创建只有一帧的 DemoData（alignment 阶段常用）
# ---------------------------------------------------------------------------

def make_static_demo(T_WE: np.ndarray, timestamp: float = 0.0) -> DemoData:
    """
    用单个位姿矩阵创建静态 demo（alignment 阶段专用）。

    Parameters
    ----------
    T_WE      : shape (4, 4)，末端执行器世界位姿
    timestamp : 时间戳 (s)，默认 0.0
    """
    return DemoData(poses=[T_WE.copy()], timestamps=[timestamp])
