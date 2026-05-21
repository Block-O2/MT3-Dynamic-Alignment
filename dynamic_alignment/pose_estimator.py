"""
MT3 动态对准扩展 — 点云位姿估计器

将分割后的物体点云转换为原始观测量 [Δx, Δy, Δθ]。

算法流程（设计笔记第八节）
--------------------------
1. 水平面过滤：剔除 |z − z_中位数| > 阈值 的竖直离群点
2. 质心计算：mean(xy) → (x_cur, y_cur)         快速、稳定
3. PCA 主轴：协方差矩阵特征分解 → θ_raw         有 180° 歧义
4. 歧义消解：与上一帧对比，选差值最小的候选方向    保持帧间连续性
5. 输出相对量：Δx = x_cur − x_ref, Δy = y_cur − y_ref, Δθ = θ_cur − θ_ref

180° 歧义处理策略
-----------------
PCA 主轴方向有 180° 不确定性（θ 和 θ+π 都是合法输出）。
从 GICP 给出的初始朝向出发，每帧选与上一帧角度差最小的候选，
利用帧间运动连续性（低速场景 < 10cm/s，每帧旋转 < 3°）消除歧义。

硬件接口
--------
实际感知依赖的 RealSense 和 MT3 分割模块均以 Stub 方式留位，
核心算法（质心 + PCA）完全不依赖硬件，可独立单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ObjectObservation


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    """点云估计器超参数"""

    min_points: int = 50
    """点云最少有效点数；低于此值认为观测无效（is_valid=False）"""

    use_pca_angle: bool = True
    """是否估计朝向角；对圆柱等旋转对称物体可设为 False"""

    z_plane_threshold: float = 0.05
    """
    水平面过滤阈值 (m)：只保留 |z − z_median| < 阈值 的点。
    典型值 0.05m，滤除背景桌面、上方遮挡点等。
    """


# ---------------------------------------------------------------------------
# PoseEstimator
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    点云 → [Δx, Δy, Δθ] 原始观测。

    使用前必须调用 initialize()，提供 GICP 对准完成时的参考点云
    和初始朝向（用于 180° 歧义消解的种子值）。

    典型使用
    --------
    ```python
    estimator = PoseEstimator()
    estimator.initialize(ref_cloud, initial_theta=gicp_theta)

    for cloud, t in stream:
        obs = estimator.estimate(cloud, timestamp=t)
        kalman.update(obs)
    ```
    """

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.cfg = config if config is not None else EstimatorConfig()

        # 参考帧（GICP 对准完成时刻）
        self._ref_centroid: np.ndarray | None = None   # shape (2,): [x_ref, y_ref]
        self._ref_theta:    float | None = None        # 参考朝向 (rad)

        # 上一帧已消歧角度（用于帧间连续性判断）
        self._prev_theta: float | None = None

    # ------------------------------------------------------------------
    # 初始化参考帧
    # ------------------------------------------------------------------

    def initialize(
        self,
        reference_cloud: np.ndarray,
        initial_theta:   float = 0.0,
    ) -> None:
        """
        设定参考状态（GICP 对准完成时刻调用一次）。

        Parameters
        ----------
        reference_cloud : 参考帧物体点云，shape (N, 3)，单位 m
        initial_theta   : GICP 给出的初始朝向 (rad)。
                          作为 180° 歧义消解的连续性追踪起点。
        """
        if reference_cloud.shape[0] < self.cfg.min_points:
            raise ValueError(
                f"参考点云点数 {reference_cloud.shape[0]} "
                f"< min_points {self.cfg.min_points}"
            )

        centroid, _ = compute_centroid_and_pca(reference_cloud)
        self._ref_centroid = centroid[:2].copy()   # 只用 x, y
        self._ref_theta    = float(initial_theta)
        self._prev_theta   = float(initial_theta)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def estimate(
        self,
        cloud:     np.ndarray,
        timestamp: float,
    ) -> ObjectObservation:
        """
        输入当前帧物体点云，输出原始观测量。

        Parameters
        ----------
        cloud     : 分割后的物体点云，shape (N, 3)，单位 m
        timestamp : 当前帧时间戳 (s)

        Returns
        -------
        ObjectObservation
            is_valid=False 时表示本帧不可用（点数不足或估计器未初始化）
        """
        if self._ref_centroid is None:
            raise RuntimeError(
                "PoseEstimator 未初始化，请先调用 initialize()"
            )

        _invalid = ObjectObservation(
            delta_x=0.0, delta_y=0.0, delta_theta=0.0,
            timestamp=timestamp, is_valid=False,
        )

        # ── 点数检查 ──
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            return _invalid
        if cloud.shape[0] < self.cfg.min_points:
            return _invalid

        # ── 水平面过滤 ──
        cloud_h = _filter_horizontal_plane(cloud, self.cfg.z_plane_threshold)
        if cloud_h.shape[0] < self.cfg.min_points:
            return _invalid

        # ── 质心 ──
        centroid, principal_axes = compute_centroid_and_pca(cloud_h)
        cur_xy = centroid[:2]

        # ── PCA 朝向 + 180° 歧义消解 ──
        if self.cfg.use_pca_angle:
            # 主轴方向（最大特征值对应，即物体最长方向）
            # principal_axes 的第 0 行是第一主轴（X-Y 平面投影角）
            ax = principal_axes[0]                 # shape (3,)
            raw_theta = float(np.arctan2(ax[1], ax[0]))
            cur_theta = _resolve_180_ambiguity(raw_theta, self._prev_theta)
            self._prev_theta = cur_theta
        else:
            # 旋转对称物体：保持上一帧角度不变
            cur_theta = self._prev_theta if self._prev_theta is not None else 0.0

        # ── 计算相对量 ──
        delta_x     = float(cur_xy[0] - self._ref_centroid[0])
        delta_y     = float(cur_xy[1] - self._ref_centroid[1])
        delta_theta = _wrap_angle(cur_theta - self._ref_theta)

        return ObjectObservation(
            delta_x=delta_x,
            delta_y=delta_y,
            delta_theta=delta_theta,
            timestamp=timestamp,
            is_valid=True,
        )

    @property
    def is_initialized(self) -> bool:
        return self._ref_centroid is not None

    # ------------------------------------------------------------------
    # Stub：硬件接口占位（不实现，仅标注接入点）
    # ------------------------------------------------------------------

    @staticmethod
    def get_point_cloud_from_realsense() -> np.ndarray:
        """
        [STUB] 从 RealSense D415 获取当前帧原始点云。

        实际部署时替换为：
        ```python
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        depth  = frames.get_depth_frame()
        pc     = rs.pointcloud()
        points = pc.calculate(depth)
        v      = np.asarray(points.get_vertices())
        return v.view(np.float32).reshape(-1, 3)
        ```
        Returns
        -------
        cloud : shape (N, 3)，单位 m
        """
        raise NotImplementedError(
            "[STUB] 请替换为实际的 RealSense 点云获取代码"
        )

    @staticmethod
    def segment_object_by_bbox(
        cloud: np.ndarray,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float,
    ) -> np.ndarray:
        """
        [STUB] 用 3D bounding box 从完整场景点云中分割物体。

        这是 MT3 现有分割能力的接入点。实际部署时替换为
        MT3 点云分割模块的调用（支持语义分割或几何分割）。

        Parameters
        ----------
        cloud   : 完整场景点云，shape (N, 3)
        x/y/z_* : bbox 边界 (m)

        Returns
        -------
        object_cloud : shape (M, 3)
        """
        raise NotImplementedError(
            "[STUB] 请接入 MT3 的点云分割模块"
        )


# ---------------------------------------------------------------------------
# 公开的算法辅助函数（可独立测试）
# ---------------------------------------------------------------------------

def compute_centroid_and_pca(
    cloud: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算点云质心和 PCA 主轴。

    Parameters
    ----------
    cloud : shape (N, 3)

    Returns
    -------
    centroid       : shape (3,)，点云质心 (x, y, z)
    principal_axes : shape (3, 3)，每行是一个主轴方向向量，
                     按特征值**降序**排列（第 0 行 = 最长轴）
    """
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid

    # 协方差矩阵（归一化为无偏估计）
    cov = (centered.T @ centered) / max(len(cloud) - 1, 1)

    # eigh 保证实对称矩阵的特征值为实数，比 eig 更稳定
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh 返回升序；反转为降序（最大特征值 = 最长轴 = 索引 0）
    order          = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, order].T   # 每行一个主轴

    return centroid, principal_axes


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _filter_horizontal_plane(
    cloud:     np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    保留 |z − z_median| ≤ threshold 的点。

    对头部固定相机（俯视）场景，物体点云的 z 值集中在某高度，
    此过滤有效去除桌面背景点和上方遮挡散点。
    """
    z_med = float(np.median(cloud[:, 2]))
    mask  = np.abs(cloud[:, 2] - z_med) <= threshold
    return cloud[mask]


def _resolve_180_ambiguity(theta: float, prev_theta: float) -> float:
    """
    消解 PCA 主轴的 180° 方向歧义。

    PCA 主轴存在符号不确定性：θ 和 θ+π 都是合法输出。
    策略：从两个候选中选与 prev_theta 角度差（绝对值）最小的一个，
    利用帧间旋转连续性（低速场景每帧旋转 < 3°）唯一确定方向。

    Parameters
    ----------
    theta      : 当前帧 PCA 原始角度（可能有 180° 翻转）
    prev_theta : 上一帧已消歧角度

    Returns
    -------
    消歧后的角度（与 prev_theta 保持连续）
    """
    candidate_a = theta
    candidate_b = theta + np.pi

    diff_a = abs(_wrap_angle(candidate_a - prev_theta))
    diff_b = abs(_wrap_angle(candidate_b - prev_theta))

    return float(candidate_a if diff_a <= diff_b else candidate_b)


def _wrap_angle(angle: float) -> float:
    """将角度归一化到 (−π, π]"""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)
