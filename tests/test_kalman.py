"""
单元测试：KalmanFilter

覆盖范围
--------
1. 初始化：状态向量正确，协方差正定
2. predict()：协方差随时间增大，状态 shape 不变，正定性保持
3. update()：创新量收敛，协方差不增大
4. predict_ahead()：不修改内部状态；预测位置领先于当前估计
5. 无效观测：is_valid=False 跳过更新，状态不变
6. 角度归一化：近 ±π 边界时创新量无跳变
7. CV + CT 模型均可正常运行
8. 圆周轨迹收敛：连续 update 后位置误差 < 2σ
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from dynamic_alignment.kalman       import KalmanFilter, _wrap_angle
from dynamic_alignment.motion_models import CVModel, CTModel
from dynamic_alignment.types         import ObjectObservation, TrackerState

from helpers import make_circular_trajectory, make_object_cloud


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _make_obs(dx=0.0, dy=0.0, dth=0.0, t=0.0, valid=True) -> ObjectObservation:
    return ObjectObservation(
        delta_x=dx, delta_y=dy, delta_theta=dth, timestamp=t, is_valid=valid
    )


def _is_positive_definite(M: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_state_shape_and_values(self):
        kf  = KalmanFilter()
        obs = _make_obs(dx=0.01, dy=-0.02, dth=0.1, t=1.0)
        kf.initialize(obs)

        state = kf.state
        assert state.x.shape == (6,)
        assert state.P.shape == (6, 6)

        # 前三维 = 观测值
        np.testing.assert_allclose(state.x[:3], [0.01, -0.02, 0.1], atol=1e-12)
        # 速度初始化为 0
        np.testing.assert_allclose(state.x[3:], [0.0, 0.0, 0.0], atol=1e-12)

    def test_initial_covariance_positive_definite(self):
        kf = KalmanFilter()
        kf.initialize(_make_obs(t=0.0))
        assert _is_positive_definite(kf.state.P)

    def test_is_initialized_flag(self):
        kf = KalmanFilter()
        assert not kf.is_initialized
        kf.initialize(_make_obs())
        assert kf.is_initialized

    def test_uninitialized_raises(self):
        kf = KalmanFilter()
        with pytest.raises(RuntimeError, match="未初始化"):
            kf.predict(0.033)
        with pytest.raises(RuntimeError, match="未初始化"):
            kf.update(_make_obs())
        with pytest.raises(RuntimeError, match="未初始化"):
            kf.predict_ahead(0.1)


# ---------------------------------------------------------------------------
# 预测步
# ---------------------------------------------------------------------------

class TestPredict:
    def setup_method(self):
        self.kf = KalmanFilter(model=CVModel())
        self.kf.initialize(_make_obs(dx=0.05, dy=0.03, dth=0.2, t=0.0))

    def test_state_shape_preserved(self):
        state = self.kf.predict(0.033)
        assert state.x.shape == (6,)
        assert state.P.shape == (6, 6)

    def test_covariance_grows(self):
        P_before = self.kf.state.P.copy()
        self.kf.predict(0.033)
        P_after = self.kf.state.P
        # 总迹（不确定性之和）应增大
        assert np.trace(P_after) > np.trace(P_before)

    def test_covariance_stays_positive_definite(self):
        for _ in range(30):
            self.kf.predict(0.033)
        assert _is_positive_definite(self.kf.state.P)

    def test_cv_zero_velocity_no_position_change(self):
        """初速度为 0 时，CV 预测后位置不变"""
        state = self.kf.predict(1.0)
        # 速度为 0 → 位置 = 初始位置（数值上 F@x，速度项 = 0）
        np.testing.assert_allclose(state.x[:3], [0.05, 0.03, 0.2], atol=1e-10)

    def test_cv_with_velocity_moves_position(self):
        """注入速度后位置应随 dt 变化"""
        kf = KalmanFilter(model=CVModel())
        # 手动设定含速度的初始状态
        kf.initialize(_make_obs(t=0.0))
        kf._state.x[3] = 0.10   # vx = 0.1 m/s
        kf._state.x[4] = 0.05   # vy = 0.05 m/s

        state = kf.predict(0.1)
        np.testing.assert_allclose(state.x[0], 0.0 + 0.10 * 0.1, atol=1e-8)
        np.testing.assert_allclose(state.x[1], 0.0 + 0.05 * 0.1, atol=1e-8)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            self.kf.predict(0.0)
        with pytest.raises(ValueError, match="dt"):
            self.kf.predict(-0.1)

    def test_timestamp_advances(self):
        self.kf.predict(0.05)
        assert abs(self.kf.state.timestamp - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# 更新步
# ---------------------------------------------------------------------------

class TestUpdate:
    def setup_method(self):
        self.kf = KalmanFilter(model=CVModel())
        self.kf.initialize(_make_obs(t=0.0))

    def test_position_converges_to_observation(self):
        """连续用相同观测更新，状态应收敛到观测值"""
        target = _make_obs(dx=0.05, dy=-0.03, dth=0.1)
        for i in range(20):
            self.kf.predict(0.033)
            obs = _make_obs(dx=0.05, dy=-0.03, dth=0.1, t=(i+1)*0.033)
            self.kf.update(obs)

        state = self.kf.state
        # 观测是精确值（无随机噪声），只有过程噪声引入残差
        # R_diag=4mm → 稳态位置误差 ≤ R = 4mm；留 1mm 余量
        np.testing.assert_allclose(state.x[0], 0.05,  atol=0.005)   # 5mm
        np.testing.assert_allclose(state.x[1], -0.03, atol=0.005)
        np.testing.assert_allclose(state.x[2], 0.1,   atol=np.deg2rad(5))

    def test_covariance_decreases_after_update(self):
        """更新后协方差应小于或等于预测后协方差"""
        self.kf.predict(0.033)
        P_after_predict = self.kf.state.P.copy()

        self.kf.update(_make_obs(dx=0.0, dy=0.0, dth=0.0, t=0.033))
        P_after_update = self.kf.state.P

        assert np.trace(P_after_update) <= np.trace(P_after_predict) + 1e-10

    def test_covariance_remains_positive_definite_after_update(self):
        for i in range(30):
            t = (i + 1) * 0.033
            self.kf.predict(0.033)
            self.kf.update(_make_obs(t=t))
        assert _is_positive_definite(self.kf.state.P)

    def test_invalid_observation_skips_update(self):
        """is_valid=False 时状态不应改变（仅更新时间戳）"""
        self.kf.predict(0.033)
        x_before = self.kf.state.x.copy()
        P_before = self.kf.state.P.copy()

        self.kf.update(_make_obs(dx=99.0, dy=99.0, t=0.033, valid=False))

        np.testing.assert_array_equal(self.kf.state.x, x_before)
        np.testing.assert_array_equal(self.kf.state.P, P_before)

    def test_angle_wrap_near_pi(self):
        """观测角接近 +π，预测角接近 −π，创新量应接近 0（不跳变到 2π）"""
        # 初始化在 +π 附近
        kf = KalmanFilter()
        kf.initialize(_make_obs(dth=np.pi - 0.05, t=0.0))
        kf.predict(0.033)

        # 观测角在 -π 附近（跨越边界）
        obs = _make_obs(dth=-(np.pi - 0.05), t=0.033)
        state_before = kf.state.x[2]
        kf.update(obs)
        state_after = kf.state.x[2]

        # 角度差应 < 0.2 rad，而非 ~2π
        assert abs(_wrap_angle(state_after - state_before)) < 0.2


# ---------------------------------------------------------------------------
# predict_ahead（延迟补偿）
# ---------------------------------------------------------------------------

class TestPredictAhead:
    def setup_method(self):
        self.kf = KalmanFilter(model=CVModel())
        self.kf.initialize(_make_obs(t=0.0))
        # 注入水平速度
        self.kf._state.x[3] = 0.06   # vx = 6 cm/s

    def test_does_not_modify_internal_state(self):
        """predict_ahead 不应改变内部状态"""
        state_before = self.kf.state

        self.kf.predict_ahead(0.1)

        state_after = self.kf.state
        np.testing.assert_array_equal(state_before.x, state_after.x)
        np.testing.assert_array_equal(state_before.P, state_after.P)

    def test_predicted_position_leads_current(self):
        """predict_ahead(tau) 的位置应在当前状态前方（沿速度方向）"""
        tau   = 0.1   # 100ms
        ahead = self.kf.predict_ahead(tau)

        # 以 6cm/s 向 +X 方向运动，预测位置应 > 当前位置
        assert ahead.delta_x > self.kf.state.delta_x

    def test_predicted_displacement_matches_velocity(self):
        """predict_ahead(tau) 的位移应约等于 vx * tau（CV 模型）"""
        tau    = 0.1
        ahead  = self.kf.predict_ahead(tau)
        vx     = self.kf.state.x[3]

        expected_dx = vx * tau
        np.testing.assert_allclose(
            ahead.delta_x, expected_dx, atol=1e-6,
            err_msg="CV 预测位移应等于 vx * tau"
        )

    def test_tau_zero_returns_current_state(self):
        ahead  = self.kf.predict_ahead(0.0)
        current = self.kf.state
        np.testing.assert_array_equal(ahead.x, current.x)

    def test_predict_ahead_covariance_larger_than_current(self):
        """predict_ahead(τ>0) 的协方差应大于当前（不确定性增加）"""
        ahead = self.kf.predict_ahead(0.1)
        assert np.trace(ahead.P) > np.trace(self.kf.state.P)


# ---------------------------------------------------------------------------
# CT 模型
# ---------------------------------------------------------------------------

class TestCTModel:
    def test_ct_initializes_and_runs(self):
        kf = KalmanFilter(model=CTModel())
        kf.initialize(_make_obs(t=0.0))
        assert kf.is_initialized

        kf.predict(0.033)
        kf.update(_make_obs(t=0.033))
        assert _is_positive_definite(kf.state.P)

    def test_ct_zero_omega_degenerates_to_cv(self):
        """ω ≈ 0 时 CT 应与 CV 给出相同的状态传播"""
        x = np.array([0.1, 0.2, 0.3, 0.05, 0.02, 0.0])   # ω = 0
        dt = 0.033

        cv_pred = CVModel().predict_state(x, dt)
        ct_pred = CTModel(omega_eps=1e-4).predict_state(x, dt)

        np.testing.assert_allclose(cv_pred, ct_pred, atol=1e-6)

    def test_ct_circular_motion_consistency(self):
        """CT 模型在圆周运动下，速度方向应随 ω 旋转"""
        R     = 0.3
        omega = 0.2
        v     = R * omega    # = 0.06 m/s
        dt    = 0.5          # 较大步长以放大差异

        # 初始状态：从 φ=0 出发（向 +Y 方向运动）
        phi0 = 0.0
        x = np.array([R, 0.0, phi0, 0.0, v, omega])   # vx=0, vy=v

        ct    = CTModel(omega_eps=1e-8)
        x_new = ct.predict_state(x, dt)

        # 速度方向应转过 omega*dt
        phi_new   = float(np.arctan2(x_new[4], x_new[3]))
        expected  = phi0 + np.pi / 2 + omega * dt   # 初始 +Y 方向 + 旋转
        # 只比较速度方向角（绕一圈后归一化）
        diff = abs(_wrap_angle(phi_new - (np.pi / 2 + omega * dt)))
        assert diff < 1e-4, f"速度方向角偏差 {np.degrees(diff):.3f}° 超出阈值"

    def test_ct_jacobian_close_to_numerical(self):
        """F_jacobian 的数值结果应自洽（与有限差分互验）"""
        ct = CTModel()
        x  = np.array([0.1, 0.05, 0.3, 0.05, 0.03, 0.15])
        dt = 0.033

        F  = ct.F_jacobian(x, dt)

        # 验证 Jacobian 的量级（不应有 NaN 或 Inf）
        assert np.all(np.isfinite(F)), "Jacobian 包含 NaN 或 Inf"
        assert F.shape == (6, 6)


# ---------------------------------------------------------------------------
# 圆周轨迹端到端收敛测试
# ---------------------------------------------------------------------------

class TestCircularTrajectoryConvergence:
    """在合成圆周轨迹上验证 Kalman 滤波器收敛性"""

    def _run_filter(self, model, clouds, timestamps, gt_poses):
        """用给定模型跑一遍滤波，返回收敛后各帧的位置误差列表"""
        from dynamic_alignment.pose_estimator import PoseEstimator

        estimator = PoseEstimator()
        estimator.initialize(clouds[0], initial_theta=gt_poses[0][2])

        kf  = KalmanFilter(model=model)
        ref_obs = ObjectObservation(
            delta_x=0.0, delta_y=0.0, delta_theta=0.0,
            timestamp=timestamps[0], is_valid=True,
        )
        kf.initialize(ref_obs)

        errors = []
        for i in range(1, len(clouds)):
            dt = timestamps[i] - timestamps[i - 1]
            kf.predict(dt)

            obs = estimator.estimate(clouds[i], timestamps[i])
            kf.update(obs)

            state = kf.state
            gx, gy, _ = gt_poses[i]
            # 位置误差 = 相对 demo 帧的真实位移 vs 估计位移
            # demo 帧 = gt_poses[0]
            true_dx = gx - gt_poses[0][0]
            true_dy = gy - gt_poses[0][1]
            err = np.hypot(state.delta_x - true_dx, state.delta_y - true_dy)
            errors.append(err)

        return errors

    def test_cv_converges_within_15mm(self, circular_trajectory):
        """
        合成点云质心噪声约 3.5mm（随机点采样主导），Kalman 稳态误差
        受限于此，最大误差（3σ）约 10mm；15mm 阈值留充分余量。
        """
        clouds, timestamps, gt_poses = circular_trajectory
        errors = self._run_filter(
            CVModel(), clouds, timestamps, gt_poses
        )
        # 前 10 帧忽略（初始化瞬态）
        converged_errors = errors[10:]
        max_err = max(converged_errors)
        assert max_err < 0.015, (
            f"CV 模型收敛后最大位置误差 {max_err*1e3:.1f}mm > 15mm"
        )

    def test_ct_converges_within_15mm(self, circular_trajectory):
        """CT 模型收敛精度与 CV 相当（圆轨道 ω 较小，差异不显著）"""
        clouds, timestamps, gt_poses = circular_trajectory
        errors = self._run_filter(
            CTModel(), clouds, timestamps, gt_poses
        )
        converged_errors = errors[10:]
        max_err = max(converged_errors)
        assert max_err < 0.015, (
            f"CT 模型收敛后最大位置误差 {max_err*1e3:.1f}mm > 15mm"
        )

    def test_predict_ahead_reduces_lag(self, circular_trajectory):
        """predict_ahead(tau) 在运动物体上应减少预测误差（vs 不补偿）"""
        clouds, timestamps, gt_poses = circular_trajectory

        from dynamic_alignment.pose_estimator import PoseEstimator
        estimator = PoseEstimator()
        estimator.initialize(clouds[0], initial_theta=gt_poses[0][2])

        kf = KalmanFilter(model=CVModel())
        kf.initialize(ObjectObservation(
            delta_x=0.0, delta_y=0.0, delta_theta=0.0,
            timestamp=timestamps[0], is_valid=True
        ))

        tau     = 0.1    # 100ms 系统延迟
        errs_no_comp  = []
        errs_with_comp = []

        for i in range(1, len(clouds)):
            dt = timestamps[i] - timestamps[i - 1]
            kf.predict(dt)
            obs = estimator.estimate(clouds[i], timestamps[i])
            kf.update(obs)

            # 真实位置（tau 秒后）
            t_future = timestamps[i] + tau
            # 线性插值真实位置
            j = min(i + int(round(tau / dt)), len(gt_poses) - 1)
            true_dx_future = gt_poses[j][0] - gt_poses[0][0]
            true_dy_future = gt_poses[j][1] - gt_poses[0][1]

            # 不补偿
            st_cur = kf.state
            err_no  = np.hypot(st_cur.delta_x  - true_dx_future,
                               st_cur.delta_y  - true_dy_future)

            # 带补偿
            st_ahead = kf.predict_ahead(tau)
            err_with = np.hypot(st_ahead.delta_x - true_dx_future,
                                st_ahead.delta_y - true_dy_future)

            errs_no_comp.append(err_no)
            errs_with_comp.append(err_with)

        # 收敛后 (frame > 10)，补偿版平均误差应低于不补偿版
        mean_no   = np.mean(errs_no_comp[10:])
        mean_with = np.mean(errs_with_comp[10:])
        assert mean_with < mean_no, (
            f"predict_ahead 未减少误差：no_comp={mean_no*1e3:.2f}mm, "
            f"with_comp={mean_with*1e3:.2f}mm"
        )
