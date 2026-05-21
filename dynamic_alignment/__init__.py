"""
dynamic_alignment — MT3 动态对准扩展模块

公开 API
--------
DynamicAlignmentTracker   主接口（init / update / get_target_pose）
CVModel / CTModel         运动模型
KalmanFilter              滤波器（可单独使用）
PoseEstimator             点云估计器（可单独使用）
ObjectObservation         观测数据结构
TrackerState              滤波器状态数据结构
DemoData / make_static_demo  demo 数据结构
state_to_transform        工具函数：状态 → 4×4 矩阵
"""

from .tracker       import DynamicAlignmentTracker, state_to_transform, transform_to_state_values
from .kalman        import KalmanFilter
from .motion_models import CVModel, CTModel, MotionModel
from .pose_estimator import PoseEstimator, EstimatorConfig, compute_centroid_and_pca
from .types         import ObjectObservation, TrackerState, DemoData, make_static_demo

__all__ = [
    # 主接口
    "DynamicAlignmentTracker",
    # 子模块
    "KalmanFilter",
    "PoseEstimator",
    "EstimatorConfig",
    # 运动模型
    "CVModel",
    "CTModel",
    "MotionModel",
    # 数据结构
    "ObjectObservation",
    "TrackerState",
    "DemoData",
    "make_static_demo",
    # 工具函数
    "state_to_transform",
    "transform_to_state_values",
    "compute_centroid_and_pca",
]
