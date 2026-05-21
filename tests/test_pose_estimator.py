"""
单元测试：PoseEstimator

覆盖范围
--------
1. compute_centroid_and_pca：质心正确；主轴正交单位向量；最大特征值在前
2. 初始化：参考质心和朝向设定正确
3. estimate()：
   - 零偏移时 delta 全为 0
   - 已知平移量时 delta_x/y 正确
   - 已知旋转量时 delta_theta 正确（含 180° 消歧测试）
4. 点数不足时返回 is_valid=False
5. 未初始化时抛出 RuntimeError
6. 水平面过滤正确剔除离群 z 点
7. 连续旋转时不出现 180° 跳变
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from dynamic_alignment.pose_estimator import (
    PoseEstimator,
    EstimatorConfig,
    compute_centroid_and_pca,
    _filter_horizontal_plane,
    _resolve_180_ambiguity,
    _wrap_angle,
)

from helpers import make_object_cloud


# ---------------------------------------------------------------------------
# compute_centroid_and_pca（纯算法，无需初始化）
# ---------------------------------------------------------------------------

class TestComputeCentroidAndPCA:
    def _make_elongated_cloud(self, angle: float, n: int = 1000) -> np.ndarray:
        """生成沿指定方向延伸的细长点云"""
        rng = np.random.default_rng(1)
        # 局部：长轴 X（0.4m），短轴 Y（0.05m）
        lx = rng.uniform(-0.2, 0.2, n)
        ly = rng.uniform(-0.025, 0.025, n)
        c, s = np.cos(angle), np.sin(angle)
        wx = c * lx - s * ly
        wy = s * lx + c * ly
        wz = rng.uniform(0, 0.03, n)
        return np.column_stack([wx, wy, wz])

    def test_centroid_at_origin(self):
        cloud = self._make_elongated_cloud(0.0)
        centroid, _ = compute_centroid_and_pca(cloud)
        np.testing.assert_allclose(centroid[:2], [0.0, 0.0], atol=0.005)

    def test_centroid_at_offset(self):
        cloud = self._make_elongated_cloud(0.0) + np.array([0.3, -0.1, 0.0])
        centroid, _ = compute_centroid_and_pca(cloud)
        np.testing.assert_allclose(centroid[:2], [0.3, -0.1], atol=0.005)

    def test_principal_axes_orthonormal(self):
        cloud = self._make_elongated_cloud(np.pi / 4)
        _, axes = compute_centroid_and_pca(cloud)
        # 每行是单位向量
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(axes[i]), 1.0, atol=1e-10)
        # 任意两行正交
        for i in range(3):
            for j in range(i + 1, 3):
                dot = abs(float(axes[i] @ axes[j]))
                assert dot < 1e-10, f"轴 {i} 与轴 {j} 非正交：dot={dot}"

    def test_principal_axis_aligned_with_long_axis(self):
        """第一主轴应与物体长轴对齐（误差 < 3°，含 180° 歧义）"""
        angle = 0.6   # rad
        cloud = self._make_elongated_cloud(angle)
        _, axes = compute_centroid_and_pca(cloud)
        ax0  = axes[0]
        est  = float(np.arctan2(ax0[1], ax0[0]))

        # 处理 180° 歧义
        diff = min(
            abs(_wrap_angle(est - angle)),
            abs(_wrap_angle(est - angle - np.pi)),
        )
        assert diff < np.deg2rad(3), (
            f"第一主轴与长轴偏差 {np.degrees(diff):.1f}° > 3°"
        )

    def test_single_point_cloud(self):
        """单点点云不应崩溃"""
        cloud = np.array([[1.0, 2.0, 3.0]])
        centroid, axes = compute_centroid_and_pca(cloud)
        assert centroid.shape == (3,)
        assert axes.shape == (3, 3)


# ---------------------------------------------------------------------------
# PoseEstimator 初始化
# ---------------------------------------------------------------------------

class TestPoseEstimatorInit:
    def test_not_initialized_raises(self):
        est = PoseEstimator()
        assert not est.is_initialized
        with pytest.raises(RuntimeError, match="未初始化"):
            est.estimate(make_object_cloud(0, 0, 0), timestamp=0.0)

    def test_initialize_sets_state(self):
        est  = PoseEstimator()
        cloud = make_object_cloud(0.3, 0.1, 0.5)
        est.initialize(cloud, initial_theta=0.5)
        assert est.is_initialized

    def test_too_few_points_raises(self):
        est  = PoseEstimator(config=EstimatorConfig(min_points=50))
        tiny = np.random.rand(10, 3) * 0.1
        with pytest.raises(ValueError, match="点数"):
            est.initialize(tiny)


# ---------------------------------------------------------------------------
# estimate() 基本功能
# ---------------------------------------------------------------------------

class TestEstimate:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.ref_cloud = make_object_cloud(0.5, 0.3, 0.0, rng=self.rng)
        self.est = PoseEstimator()
        self.est.initialize(self.ref_cloud, initial_theta=0.0)

    def test_same_cloud_zero_delta(self):
        """同一位置估计应返回接近 0 的 delta"""
        obs = self.est.estimate(self.ref_cloud, timestamp=0.1)
        assert obs.is_valid
        assert abs(obs.delta_x)     < 0.005   # 5mm
        assert abs(obs.delta_y)     < 0.005
        assert abs(obs.delta_theta) < np.deg2rad(3)

    def test_translated_cloud_delta_x(self):
        """X 方向平移 5cm，delta_x 应接近 0.05m"""
        shifted = self.ref_cloud.copy()
        shifted[:, 0] += 0.05
        obs = self.est.estimate(shifted, timestamp=0.1)
        assert obs.is_valid
        np.testing.assert_allclose(obs.delta_x, 0.05, atol=0.003)
        np.testing.assert_allclose(obs.delta_y, 0.0,  atol=0.003)

    def test_translated_cloud_delta_y(self):
        """Y 方向平移 -3cm，delta_y 应接近 -0.03m"""
        shifted = self.ref_cloud.copy()
        shifted[:, 1] -= 0.03
        obs = self.est.estimate(shifted, timestamp=0.1)
        assert obs.is_valid
        np.testing.assert_allclose(obs.delta_y, -0.03, atol=0.003)

    def test_rotated_cloud_delta_theta(self):
        """旋转 20° 后 delta_theta 应接近 0.35 rad（20°）"""
        target_angle = np.deg2rad(20)
        cloud = make_object_cloud(0.5, 0.3, target_angle, rng=self.rng)
        obs = self.est.estimate(cloud, timestamp=0.1)
        assert obs.is_valid
        # 允许误差 5°
        assert abs(_wrap_angle(obs.delta_theta - target_angle)) < np.deg2rad(5), (
            f"delta_theta={np.degrees(obs.delta_theta):.1f}° vs "
            f"expected={np.degrees(target_angle):.1f}°"
        )

    def test_invalid_when_too_few_points(self):
        tiny = np.random.rand(5, 3) * 0.1
        obs = self.est.estimate(tiny, timestamp=0.1)
        assert not obs.is_valid

    def test_invalid_when_wrong_shape(self):
        bad = np.random.rand(100, 2)   # 缺 z 列
        obs = self.est.estimate(bad, timestamp=0.1)
        assert not obs.is_valid

    def test_timestamp_in_observation(self):
        obs = self.est.estimate(self.ref_cloud, timestamp=3.14)
        assert abs(obs.timestamp - 3.14) < 1e-9


# ---------------------------------------------------------------------------
# 水平面过滤
# ---------------------------------------------------------------------------

class TestHorizontalFilter:
    def test_removes_high_z_outliers(self):
        rng = np.random.default_rng(7)
        # 正常点 z ≈ 0.02
        normal = rng.uniform(-0.1, 0.1, (200, 3))
        normal[:, 2] = rng.uniform(0.0, 0.04, 200)
        # 离群点 z = 0.5
        outliers = rng.uniform(-0.1, 0.1, (20, 3))
        outliers[:, 2] = 0.5
        cloud = np.vstack([normal, outliers])

        filtered = _filter_horizontal_plane(cloud, threshold=0.05)
        assert filtered.shape[0] <= 200 + 1   # 离群点应被剔除
        assert np.max(np.abs(filtered[:, 2] - np.median(filtered[:, 2]))) <= 0.05


# ---------------------------------------------------------------------------
# 180° 歧义消解
# ---------------------------------------------------------------------------

class TestAmbiguityResolution:
    def test_selects_closer_candidate(self):
        prev = 0.3
        # 两个候选：0.35（近）和 0.35+π（远）
        result = _resolve_180_ambiguity(0.35, prev)
        assert abs(result - 0.35) < abs(result - (0.35 + np.pi))

    def test_selects_pi_candidate_when_closer(self):
        prev = np.pi - 0.1
        # θ = 0.05（远），θ+π ≈ π+0.05（近）
        result = _resolve_180_ambiguity(0.05, prev)
        np.testing.assert_allclose(result, 0.05 + np.pi, atol=1e-10)

    def test_no_jump_in_continuous_rotation(self):
        """连续旋转场景：每帧增加 5°，不应出现 180° 跳变"""
        est = PoseEstimator()
        rng = np.random.default_rng(99)
        ref_cloud = make_object_cloud(0.0, 0.0, 0.0, rng=rng)
        est.initialize(ref_cloud, initial_theta=0.0)

        prev_theta = None
        angles = np.linspace(0, np.pi, 36)   # 0° → 180°，每步 5°

        for angle in angles:
            cloud = make_object_cloud(0.0, 0.0, angle, rng=rng)
            obs   = est.estimate(cloud, timestamp=float(angle))

            if prev_theta is not None:
                jump = abs(_wrap_angle(
                    (obs.delta_theta + est._ref_theta) - prev_theta
                ))
                assert jump < np.deg2rad(15), (
                    f"角度跳变 {np.degrees(jump):.1f}° > 15° "
                    f"(在 angle={np.degrees(angle):.0f}°)"
                )
            prev_theta = obs.delta_theta + est._ref_theta
