"""
MT3 动态对准扩展 — 主接口

DynamicAlignmentTracker 对外暴露三个方法：

    init(reference_cloud, initial_theta, timestamp)
        ── GICP 对准完成后调用一次，设定参考状态

    update(cloud, timestamp) -> TrackerState
        ── 每帧调用，执行 predict → estimate → update 完整流程

    get_target_pose(demo_data, t_demo, tau) -> np.ndarray (4×4)
        ── 返回当前控制周期的末端目标位姿

核心公式（设计笔记第六节）
--------------------------
    T_WE_target(t) = T_δ(t + τ) · T_WE_demo(t)

    Alignment 阶段：T_WE_demo(t) = T_WE_demo(0)（常数，demo 第 0 帧）
    Interaction 阶段：T_WE_demo(t) 沿录制序列推进
    切换时刻：两项目标相等，连续无跳变

T_δ 的 4×4 形式（水平面 SE(2) 嵌入 SE(3)）
-------------------------------------------
    ┌  cos(Δθ)  −sin(Δθ)  0  Δx ┐
    │  sin(Δθ)   cos(Δθ)  0  Δy │
    │     0         0     1   0  │
    └     0         0     0   1  ┘
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .types import ObjectObservation, TrackerState, DemoData
from .motion_models import MotionModel, CVModel
from .kalman import KalmanFilter
from .pose_estimator import PoseEstimator, EstimatorConfig


class DynamicAlignmentTracker:
    """
    MT3 动态对准追踪器——系统主入口。

    内部维护 KalmanFilter 和 PoseEstimator 两个子模块，
    对上层（控制器）只暴露三个方法。

    Parameters
    ----------
    tau              : 系统总延迟 (s)，默认 0.1s（感知+计算+执行）
    motion_model     : 运动模型，默认 CVModel（对延迟补偿已充分）
    estimator_config : 点云估计器配置，None 时使用默认配置
    kalman_R_diag    : Kalman 观测噪声标准差 [σ_x, σ_y, σ_θ]，
                       None 时使用 [4mm, 4mm, 3°]
    init_vel_cov     : Kalman 速度初始方差，默认 0.01 m²/s²

    典型使用
    --------
    ```python
    tracker = DynamicAlignmentTracker(tau=0.1)

    # GICP 对准完成后初始化一次
    tracker.init(ref_cloud, initial_theta=theta_gicp, timestamp=t0)

    # 控制循环
    while True:
        cloud = perception.get_object_cloud()
        t     = time.monotonic()

        state   = tracker.update(cloud, timestamp=t)
        T_target = tracker.get_target_pose(demo_data, t_demo=phase_time)
        robot.set_cartesian_target(T_target)
    ```
    """

    def __init__(
        self,
        tau:              float = 0.1,
        motion_model:     MotionModel | None = None,
        estimator_config: EstimatorConfig | None = None,
        kalman_R_diag:    np.ndarray | None = None,
        init_vel_cov:     float = 0.01,
    ) -> None:
        self.tau = float(tau)

        model = motion_model if motion_model is not None else CVModel()
        self._kf        = KalmanFilter(
            model=model,
            R_diag=kalman_R_diag,
            init_vel_cov=init_vel_cov,
        )
        self._estimator = PoseEstimator(config=estimator_config)

        self._last_timestamp: float | None = None
        self._initialized:    bool         = False
        self._adaptive_total_frames: int = 0
        self._adaptive_advanced_frames: int = 0
        self._adaptive_progress_sum: float = 0.0
        self._adaptive_last_t_demo: float | None = None

    # ------------------------------------------------------------------
    # 初始化（GICP 对准完成后调用一次）
    # ------------------------------------------------------------------

    def init(
        self,
        reference_cloud: np.ndarray,
        initial_theta:   float,
        timestamp:       float,
    ) -> None:
        """
        用 GICP 对准结果初始化追踪器。

        GICP 完成时，物体在 demo 参考位置，T_δ = I（零位移零旋转）。
        本方法将此状态作为 Kalman 初始状态写入，并设定 PoseEstimator
        的参考帧（用于后续每帧计算相对量）。

        Parameters
        ----------
        reference_cloud : GICP 完成时的物体点云，shape (N, 3)
        initial_theta   : GICP 给出的初始朝向 (rad)，用于消歧种子
        timestamp       : 初始化时刻时间戳 (s)
        """
        # 设定 PoseEstimator 参考帧
        self._estimator.initialize(
            reference_cloud=reference_cloud,
            initial_theta=initial_theta,
        )

        # 初始化 Kalman（T_δ = 0，即物体在 demo 对准位置）
        initial_obs = ObjectObservation(
            delta_x=0.0,
            delta_y=0.0,
            delta_theta=0.0,
            timestamp=timestamp,
            is_valid=True,
        )
        self._kf.initialize(initial_obs)

        self._last_timestamp = float(timestamp)
        self._initialized    = True
        self._adaptive_total_frames = 0
        self._adaptive_advanced_frames = 0
        self._adaptive_progress_sum = 0.0
        self._adaptive_last_t_demo = None

    # ------------------------------------------------------------------
    # 每帧更新
    # ------------------------------------------------------------------

    def update(self, cloud: np.ndarray, timestamp: float) -> TrackerState:
        """
        输入当前帧点云，执行完整的 predict → estimate → update 流程。

        Steps
        -----
        1. dt = timestamp − last_timestamp
        2. KalmanFilter.predict(dt)         时间推进
        3. PoseEstimator.estimate(cloud)    点云 → [Δx, Δy, Δθ] 原始观测
        4. KalmanFilter.update(obs)         融合观测

        Parameters
        ----------
        cloud     : 当前帧物体点云，shape (N, 3)
        timestamp : 当前帧时间戳 (s)，必须单调递增

        Returns
        -------
        TrackerState : 融合本帧观测后的最优状态估计
        """
        if not self._initialized:
            raise RuntimeError("Tracker 未初始化，请先调用 init()")

        dt = float(timestamp) - float(self._last_timestamp)
        if dt <= 0:
            raise ValueError(
                f"时间戳必须单调递增：上一帧 {self._last_timestamp:.4f}s，"
                f"当前帧 {timestamp:.4f}s，dt={dt:.6f}s"
            )

        # 1. 预测步
        self._kf.predict(dt)

        # 2. 点云 → 原始观测
        obs = self._estimator.estimate(cloud, timestamp=float(timestamp))

        # 3. 更新步
        state = self._kf.update(obs)

        self._last_timestamp = float(timestamp)
        return state

    # ------------------------------------------------------------------
    # 合成目标位姿（核心公式）
    # ------------------------------------------------------------------

    def get_target_pose(
        self,
        demo_data: DemoData,
        t_demo:    float,
        tau:       float | None = None,
    ) -> np.ndarray:
        """
        合成当前控制周期的末端目标位姿：

            T_WE_target = T_δ(t + τ) · T_WE_demo(t_demo)

        两阶段行为
        ----------
        Alignment（对准）阶段：传入 t_demo=0，使用 demo 第 0 帧，
            T_WE_target 随 T_δ 变化，让末端跟随物体运动。
        Interaction（交互）阶段：t_demo 随 demo 推进，
            T_WE_demo(t) 沿录制轨迹变化，T_WE_target = T_δ·T_WE_demo(t)，
            末端在物体坐标系内重放 demo 动作。
        切换时刻：两阶段目标相等（均为 T_δ·T_WE_demo(0)），连续无跳变。

        Parameters
        ----------
        demo_data : DemoData，包含 demo 位姿序列
        t_demo    : 当前 demo 时刻 (s)；
                    Alignment 阶段固定传 0，Interaction 阶段传递经过时间
        tau       : 延迟补偿量 (s)；None 时使用 self.tau

        Returns
        -------
        T_WE_target : shape (4, 4)，SE(3) 齐次变换矩阵
        """
        if not self._initialized:
            raise RuntimeError("Tracker 未初始化，请先调用 init()")

        effective_tau = tau if tau is not None else self.tau

        # τ 秒后的预测 T_δ（不修改内部状态）
        predicted_state = self._kf.predict_ahead(effective_tau)
        T_delta         = state_to_transform(predicted_state)

        # demo 参考位姿
        T_demo = demo_data.get_pose_at(float(t_demo))

        # 核心公式
        return T_delta @ T_demo

    def get_target_pose_adaptive(
        self,
        demo_data: DemoData,
        t_demo,
        lateral_error_mm: float,
        threshold_mm: float = 15.0,
        tau: float | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Adaptive demo replay that couples tracking quality with demo progression.

        Demo progression is slowed continuously as lateral error grows:

            progress_rate = clip(1 - (error_mm - threshold_mm) / 20, 0, 1)

        With the default threshold, error < 15mm advances at full speed,
        25mm advances at half speed, and >=35mm freezes progression.

        Args:
            demo_data: DemoData object.
            t_demo: mutable demo timestamp container, e.g. [t]. The caller should
                propose the next demo timestamp in t_demo[0]; this method writes
                back the adapted timestamp after applying progress_rate.
            lateral_error_mm: current lateral tracking error in mm.
            threshold_mm: max allowed error before freezing demo progression.
            tau: prediction horizon. None uses self.tau.

        Returns:
            T_WE_target: 4x4 target pose.
            demo_advanced: whether demo timestamp advanced by a nonzero amount.
        """
        if not self._initialized:
            raise RuntimeError("Tracker 未初始化，请先调用 init()")

        if not hasattr(t_demo, "__getitem__") or not hasattr(t_demo, "__setitem__"):
            raise TypeError(
                "t_demo must be a mutable timestamp container, e.g. [current_t_demo]"
            )

        self._adaptive_total_frames += 1
        proposed_t_demo = float(t_demo[0])
        lateral_error_mm = float(lateral_error_mm)
        threshold_mm = float(threshold_mm)

        progress_rate = 1.0 - (lateral_error_mm - threshold_mm) / 20.0
        progress_rate = float(np.clip(progress_rate, 0.0, 1.0))
        previous_t_demo = (
            self._adaptive_last_t_demo
            if self._adaptive_last_t_demo is not None
            else float(demo_data.timestamps[0])
        )
        proposed_dt = max(0.0, proposed_t_demo - previous_t_demo)
        active_t_demo = previous_t_demo + proposed_dt * progress_rate
        t_demo[0] = active_t_demo

        self._adaptive_progress_sum += progress_rate
        demo_advanced = active_t_demo > previous_t_demo
        if demo_advanced:
            self._adaptive_advanced_frames += 1
        self._adaptive_last_t_demo = active_t_demo

        target_pose = self.get_target_pose(
            demo_data=demo_data,
            t_demo=active_t_demo,
            tau=tau,
        )
        return target_pose, demo_advanced

    # ------------------------------------------------------------------
    # 调试 / 集成辅助
    # ------------------------------------------------------------------

    def get_T_delta(self, tau: float = 0.0) -> np.ndarray:
        """
        返回 T_δ（4×4），可选地包含 tau 秒预测。

        用于调试或直接与外部控制器集成。
        """
        if not self._initialized:
            raise RuntimeError("Tracker 未初始化，请先调用 init()")

        if tau > 0.0:
            state = self._kf.predict_ahead(tau)
        else:
            state = self._kf.state
        return state_to_transform(state)

    @property
    def current_state(self) -> TrackerState | None:
        """当前 Kalman 状态的只读副本"""
        return self._kf.state

    @property
    def progress_rate(self) -> float:
        """Mean adaptive demo progress rate across calls."""
        if self._adaptive_total_frames == 0:
            return 0.0
        return self._adaptive_progress_sum / self._adaptive_total_frames

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return (
            f"DynamicAlignmentTracker("
            f"tau={self.tau}s, "
            f"model={self._kf.model!r}, "
            f"{status})"
        )


# ---------------------------------------------------------------------------
# 辅助函数（对外公开，方便单元测试和调试）
# ---------------------------------------------------------------------------

def state_to_transform(state: TrackerState) -> np.ndarray:
    """
    将追踪状态 [Δx, Δy, Δθ, ...] 转换为 4×4 齐次变换矩阵 T_δ。

    T_δ 表示水平面刚体变换（SE(2) 嵌入 SE(3))：
    - 旋转：绕世界坐标系 Z 轴转 Δθ
    - 平移：[Δx, Δy, 0]（不改变高度）

    变换矩阵
    --------
        ┌  cos(Δθ)  −sin(Δθ)  0  Δx ┐
        │  sin(Δθ)   cos(Δθ)  0  Δy │
        │     0         0     1   0  │
        └     0         0     0   1  ┘

    Parameters
    ----------
    state : TrackerState

    Returns
    -------
    T : shape (4, 4)
    """
    dx  = state.delta_x
    dy  = state.delta_y
    dth = state.delta_theta

    c, s = np.cos(dth), np.sin(dth)

    T = np.eye(4)
    T[0, 0] =  c;  T[0, 1] = -s;  T[0, 3] = dx
    T[1, 0] =  s;  T[1, 1] =  c;  T[1, 3] = dy

    return T


def transform_to_state_values(T: np.ndarray) -> tuple[float, float, float]:
    """
    从 4×4 变换矩阵提取 (Δx, Δy, Δθ)，用于验证和调试。

    Parameters
    ----------
    T : shape (4, 4)

    Returns
    -------
    (delta_x, delta_y, delta_theta)
    """
    dx  = float(T[0, 3])
    dy  = float(T[1, 3])
    dth = float(np.arctan2(T[1, 0], T[0, 0]))
    return dx, dy, dth
