"""
examples/simulate_and_plot.py
==============================
MT3 动态对准扩展 — 合成数据验证与可视化

生成三张实验图，保存到 results/ 目录：
  results/tracking_comparison.png   三种运动模式追踪对比
  results/latency_compensation.png  延迟补偿效果
  results/convergence.png           滤波器冷启动收敛过程

运行方式（从项目根目录）：
  conda activate mt3_plus
  python examples/simulate_and_plot.py

依赖：numpy, matplotlib, seaborn（均通过 conda env mt3_plus 提供）
无硬件要求，纯合成数据。
"""

from __future__ import annotations

import sys
import os

# ── 确保能 import dynamic_alignment（无论从哪里运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 非交互后端，适合脚本和服务器
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from dynamic_alignment.kalman import KalmanFilter, _wrap_angle
from dynamic_alignment.motion_models import CVModel
from dynamic_alignment.types import ObjectObservation

# ════════════════════════════════════════════════════════════════════════════
# 全局参数
# ════════════════════════════════════════════════════════════════════════════

FPS       = 30                      # 相机帧率 Hz
DT        = 1.0 / FPS               # 时间步长 s
DURATION  = 15.0                    # 仿真时长 s
N_FRAMES  = int(DURATION * FPS)     # 450 帧

SIGMA_POS = 0.003                   # 位置观测噪声 σ = 3 mm
SIGMA_YAW = np.deg2rad(2.0)         # 角度观测噪声 σ = 2°

SPEED     = 0.05                    # 物体速度 = 5 cm/s
RADIUS    = 0.15                    # 圆轨道半径 = 15 cm
OMEGA     = SPEED / RADIUS          # 圆周角速度 ≈ 0.333 rad/s

TAU       = 0.10                    # 系统延迟 τ = 100 ms
TAU_FRAMES = int(round(TAU / DT))   # 延迟对应帧数 = 3

SEED      = 42

OUT_DIR   = Path(_ROOT) / "results"
OUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 图表样式
# ════════════════════════════════════════════════════════════════════════════

def _setup_style() -> None:
    """设置全局 matplotlib 样式（兼容新旧 seaborn）"""
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            pass
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         11,
        "axes.labelsize":    12,
        "axes.titlesize":    13,
        "legend.fontsize":   10,
        "lines.linewidth":   2.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.4,
    })

# ════════════════════════════════════════════════════════════════════════════
# 轨迹生成器
# ════════════════════════════════════════════════════════════════════════════

def gen_linear(n: int, dt: float, speed: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    匀速直线：沿 +X 方向，速度 speed (m/s)。
    返回 shape (n, 3)：[Δx, Δy, Δθ] 均相对 t=0 参考帧。
    """
    ts = np.arange(n) * dt
    dx = speed * ts
    dy = np.zeros(n)
    dth = np.zeros(n)
    return np.column_stack([dx, dy, dth])


def gen_circular(n: int, dt: float, speed: float, radius: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    匀速圆周：半径 radius (m)，线速度 speed (m/s)。
    物体朝向随速度方向旋转（如传送带弯道）。
    """
    omega = speed / radius
    ts = np.arange(n) * dt
    dx  = radius * (np.cos(omega * ts) - 1.0)
    dy  = radius * np.sin(omega * ts)
    dth = omega * ts
    return np.column_stack([dx, dy, dth])


def gen_random_slow(n: int, dt: float, speed: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    慢速随机游走：方向角做 AR(1) 漫步，速度幅值保持 speed (m/s)。
    轨迹平滑（方向相关时间 ~3s），物体不旋转（Δθ = 0）。
    """
    # 方向角慢漂移：每步随机扰动 σ=0.08 rad，时间相关性 ~3s
    angle_increments = rng.normal(0.0, 0.08 * np.sqrt(dt), n)
    angles = np.cumsum(angle_increments)                 # 累积方向角
    vx = speed * np.cos(angles)
    vy = speed * np.sin(angles)

    dx  = np.cumsum(vx * dt)
    dy  = np.cumsum(vy * dt)
    dth = np.zeros(n)                                   # 桌面平移，不旋转
    return np.column_stack([dx, dy, dth])


# ════════════════════════════════════════════════════════════════════════════
# 仿真核心：加噪声 + 运行 Kalman
# ════════════════════════════════════════════════════════════════════════════

def add_sensor_noise(true_deltas: np.ndarray,
                     sigma_pos: float,
                     sigma_yaw: float,
                     rng: np.random.Generator) -> np.ndarray:
    """在真实 delta 上叠加模拟 D415 传感器噪声"""
    n = len(true_deltas)
    obs = true_deltas.copy()
    obs[:, 0] += rng.normal(0.0, sigma_pos, n)
    obs[:, 1] += rng.normal(0.0, sigma_pos, n)
    obs[:, 2] += rng.normal(0.0, sigma_yaw, n)
    return obs


def run_kalman(true_deltas: np.ndarray,
               noisy_obs:   np.ndarray,
               dt:          float,
               model=None) -> tuple[np.ndarray, np.ndarray]:
    """
    在合成轨迹上运行 Kalman 滤波器。

    Parameters
    ----------
    true_deltas : (N, 3) 真实 [Δx, Δy, Δθ]
    noisy_obs   : (N, 3) 带噪声观测
    dt          : 时间步长 (s)
    model       : MotionModel，默认 CVModel()

    Returns
    -------
    estimates   : (N, 6) 每帧状态估计 [Δx,Δy,Δθ,Δẋ,Δẏ,Δθ̇]
    covariances : (N, 6, 6) 每帧协方差矩阵
    """
    if model is None:
        model = CVModel()

    kf = KalmanFilter(model=model)
    kf.initialize(ObjectObservation(
        delta_x=noisy_obs[0, 0],
        delta_y=noisy_obs[0, 1],
        delta_theta=noisy_obs[0, 2],
        timestamp=0.0,
    ))

    n = len(true_deltas)
    estimates   = np.zeros((n, 6))
    covariances = np.zeros((n, 6, 6))
    estimates[0]   = kf.state.x
    covariances[0] = kf.state.P

    for i in range(1, n):
        kf.predict(dt)
        kf.update(ObjectObservation(
            delta_x=noisy_obs[i, 0],
            delta_y=noisy_obs[i, 1],
            delta_theta=noisy_obs[i, 2],
            timestamp=i * dt,
        ))
        estimates[i]   = kf.state.x
        covariances[i] = kf.state.P

    return estimates, covariances


def run_kalman_with_ahead(true_deltas: np.ndarray,
                          noisy_obs:   np.ndarray,
                          dt:          float,
                          tau:         float) -> tuple[np.ndarray, np.ndarray]:
    """
    运行 Kalman 并在每帧同时记录当前估计和 predict_ahead(τ)。

    Returns
    -------
    est_now   : (N, 2) 当前时刻 XY 估计
    est_ahead : (N, 2) τ 秒后预测 XY
    """
    kf = KalmanFilter(model=CVModel())
    kf.initialize(ObjectObservation(
        delta_x=noisy_obs[0, 0],
        delta_y=noisy_obs[0, 1],
        delta_theta=noisy_obs[0, 2],
        timestamp=0.0,
    ))

    n = len(true_deltas)
    est_now   = np.zeros((n, 2))
    est_ahead = np.zeros((n, 2))

    est_now[0]   = kf.state.x[:2]
    est_ahead[0] = kf.predict_ahead(tau).x[:2]

    for i in range(1, n):
        kf.predict(dt)
        kf.update(ObjectObservation(
            delta_x=noisy_obs[i, 0],
            delta_y=noisy_obs[i, 1],
            delta_theta=noisy_obs[i, 2],
            timestamp=i * dt,
        ))
        est_now[i]   = kf.state.x[:2]
        est_ahead[i] = kf.predict_ahead(tau).x[:2]

    return est_now, est_ahead


def pos_rmse_mm(est: np.ndarray, truth: np.ndarray) -> float:
    """XY 位置 RMSE，单位 mm"""
    errs = np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])
    return float(np.sqrt(np.mean(errs ** 2))) * 1e3


def _smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    """简单矩形窗滑动平均"""
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def _rolling_std(x: np.ndarray, w: int = 30) -> np.ndarray:
    """滚动标准差（用于阴影误差带）"""
    half = w // 2
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = float(np.std(x[lo:hi]))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 图 1：追踪对比（三种运动模式）
# ════════════════════════════════════════════════════════════════════════════

def plot_tracking_comparison(out_path: Path) -> None:
    """
    三个子图并排：Linear / Circular / Random Slow Walk
    每图：真实轨迹（实线）、Kalman 估计（虚线）、噪声观测（散点）
    标注：RMSE
    """
    rng = np.random.default_rng(SEED)

    scenarios = [
        ("Linear\n(constant velocity)",
         gen_linear(N_FRAMES, DT, SPEED, rng)),
        (f"Circular\n(R = {RADIUS*100:.0f} cm, ω = {OMEGA:.2f} rad/s)",
         gen_circular(N_FRAMES, DT, SPEED, RADIUS, rng)),
        ("Random Slow Walk\n(AR(1) direction process)",
         gen_random_slow(N_FRAMES, DT, SPEED, rng)),
    ]

    # 颜色方案
    C_TRUE  = "#1565C0"   # 深蓝：真实轨迹
    C_EST   = "#E65100"   # 深橙：Kalman 估计
    C_OBS   = "#78909C"   # 蓝灰：噪声观测

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax, (title, true_deltas) in zip(axes, scenarios):
        noisy_obs         = add_sensor_noise(true_deltas, SIGMA_POS, SIGMA_YAW, rng)
        estimates, _      = run_kalman(true_deltas, noisy_obs, DT)
        rmse              = pos_rmse_mm(estimates, true_deltas)

        # ── 噪声观测（每 4 帧采样一次，降低视觉密度）──────────────────
        s = 4
        ax.scatter(
            noisy_obs[::s, 0] * 100,
            noisy_obs[::s, 1] * 100,
            s=7, alpha=0.30, color=C_OBS,
            label="Noisy observations", zorder=1, linewidths=0,
        )

        # ── 真实轨迹 ──────────────────────────────────────────────────
        ax.plot(
            true_deltas[:, 0] * 100,
            true_deltas[:, 1] * 100,
            color=C_TRUE, lw=2.2, ls="-",
            label="Ground truth", zorder=3,
        )

        # ── Kalman 估计 ───────────────────────────────────────────────
        ax.plot(
            estimates[:, 0] * 100,
            estimates[:, 1] * 100,
            color=C_EST, lw=1.8, ls="--",
            label="Kalman estimate", zorder=4,
        )

        # 起点 / 终点标记
        ax.plot(*[true_deltas[0,  j] * 100 for j in (0, 1)],
                "o", color=C_TRUE, ms=8, zorder=5, label="_nolegend_")
        ax.plot(*[true_deltas[-1, j] * 100 for j in (0, 1)],
                "s", color=C_TRUE, ms=8, zorder=5, label="_nolegend_")

        ax.set_title(title, pad=8)
        ax.set_xlabel("Δx (cm)")
        ax.set_ylabel("Δy (cm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", framealpha=0.85, fontsize=9)

        # RMSE 文字框
        ax.text(
            0.97, 0.04,
            f"RMSE = {rmse:.1f} mm",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#90A4AE", alpha=0.90),
        )

    fig.suptitle(
        f"Kalman Filter Tracking — Three Motion Modes  "
        f"(v = {SPEED*100:.0f} cm/s,  σ_pos = {SIGMA_POS*1e3:.0f} mm,  "
        f"σ_yaw = {np.degrees(SIGMA_YAW):.0f}°,  30 Hz,  15 s)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Fig 1] {out_path.name}  saved")


# ════════════════════════════════════════════════════════════════════════════
# 图 2：延迟补偿效果
# ════════════════════════════════════════════════════════════════════════════

def plot_latency_compensation(out_path: Path) -> None:
    """
    对比两条误差曲线：
      - 无预测：控制器用当前估计，在 τ 后执行时物体已移走
      - predict_ahead(τ)：预测 τ 秒后位置，与物体实际位置对齐

    使用圆周轨迹（有持续速度，延迟效果最显著）。
    """
    rng = np.random.default_rng(SEED + 10)

    true_deltas = gen_circular(N_FRAMES, DT, SPEED, RADIUS, rng)
    noisy_obs   = add_sensor_noise(true_deltas, SIGMA_POS, SIGMA_YAW, rng)

    est_now, est_ahead = run_kalman_with_ahead(true_deltas, noisy_obs, DT, TAU)

    # 只比较有未来真值的帧（去掉末尾 tau_frames 帧）
    valid      = N_FRAMES - TAU_FRAMES
    ts         = np.arange(valid) * DT

    # 误差：控制命令 vs. τ 后的真实位置（单位 mm）
    err_no = np.array([
        np.hypot(est_now[i, 0]   - true_deltas[i + TAU_FRAMES, 0],
                 est_now[i, 1]   - true_deltas[i + TAU_FRAMES, 1])
        for i in range(valid)
    ]) * 1e3

    err_with = np.array([
        np.hypot(est_ahead[i, 0] - true_deltas[i + TAU_FRAMES, 0],
                 est_ahead[i, 1] - true_deltas[i + TAU_FRAMES, 1])
        for i in range(valid)
    ]) * 1e3

    # 跳过初始化瞬态（前 0.5 s = 15 帧）
    skip = 15
    ts_p = ts[skip:]
    en   = err_no[skip:]
    ew   = err_with[skip:]

    # 平滑 + 误差带
    en_sm  = _smooth(en,  w=20)
    ew_sm  = _smooth(ew,  w=20)
    en_std = _rolling_std(en,  w=45)
    ew_std = _rolling_std(ew,  w=45)

    mean_no   = float(np.mean(en))
    mean_with = float(np.mean(ew))
    reduction = (1 - mean_with / mean_no) * 100

    C_NO   = "#E53935"    # 红：无预测
    C_WITH = "#1565C0"    # 蓝：有预测

    fig, ax = plt.subplots(figsize=(11, 5))

    # 无预测
    ax.fill_between(ts_p,
                    np.clip(en_sm - en_std, 0, None),
                    en_sm + en_std,
                    alpha=0.18, color=C_NO)
    ax.plot(ts_p, en_sm, color=C_NO, lw=2.2,
            label=f"No prediction      mean = {mean_no:.1f} mm")

    # predict_ahead(τ)
    ax.fill_between(ts_p,
                    np.clip(ew_sm - ew_std, 0, None),
                    ew_sm + ew_std,
                    alpha=0.18, color=C_WITH)
    ax.plot(ts_p, ew_sm, color=C_WITH, lw=2.2,
            label=f"predict_ahead({TAU*1000:.0f} ms)  mean = {mean_with:.1f} mm")

    # 传感器噪声基准线
    ax.axhline(SIGMA_POS * 1e3, ls=":", lw=1.5, color="#546E7A",
               label=f"Sensor noise floor  ({SIGMA_POS*1e3:.0f} mm)")

    # 标注改善比例
    ax.text(0.98, 0.94,
            f"Error reduction: {reduction:.0f} %",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#0D47A1",
            bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD",
                      ec="#1565C0", alpha=0.9))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error at execution time (mm)")
    ax.set_title(
        f"Latency Compensation Effect  —  predict_ahead(τ = {TAU*1000:.0f} ms)\n"
        f"Circular trajectory, R = {RADIUS*100:.0f} cm, "
        f"v = {SPEED*100:.0f} cm/s  →  kinematic lag ≈ {SPEED*TAU*1e3:.0f} mm",
        fontsize=12,
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(ts_p[0], ts_p[-1])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Fig 2] {out_path.name}  saved")


# ════════════════════════════════════════════════════════════════════════════
# 图 3：冷启动收敛过程
# ════════════════════════════════════════════════════════════════════════════

def plot_convergence(out_path: Path) -> None:
    """
    展示 Kalman 滤波器从冷启动到稳态的协方差衰减过程。

    上图：位置不确定度（σ_x, σ_y，单位 mm）
    下图：速度不确定度（σ_vx, σ_vy，单位 mm/s）
    """
    rng = np.random.default_rng(SEED + 20)

    # 只需要 5 秒数据来展示收敛（150 帧）
    n_conv      = int(5.0 * FPS)
    true_deltas = gen_circular(n_conv, DT, SPEED, RADIUS, rng)
    noisy_obs   = add_sensor_noise(true_deltas, SIGMA_POS, SIGMA_YAW, rng)

    # ── 运行 Kalman，逐帧记录协方差对角元 ─────────────────────────────
    kf = KalmanFilter(model=CVModel())
    kf.initialize(ObjectObservation(
        delta_x=noisy_obs[0, 0],
        delta_y=noisy_obs[0, 1],
        delta_theta=noisy_obs[0, 2],
        timestamp=0.0,
    ))

    frames  = np.arange(n_conv)
    sig_x   = np.zeros(n_conv)   # √P[0,0]  mm
    sig_y   = np.zeros(n_conv)   # √P[1,1]  mm
    sig_vx  = np.zeros(n_conv)   # √P[3,3]  mm/s
    sig_vy  = np.zeros(n_conv)   # √P[4,4]  mm/s

    def _read_P(kf_):
        P = kf_.state.P
        return (
            np.sqrt(P[0, 0]) * 1e3,
            np.sqrt(P[1, 1]) * 1e3,
            np.sqrt(P[3, 3]) * 1e3,
            np.sqrt(P[4, 4]) * 1e3,
        )

    sig_x[0], sig_y[0], sig_vx[0], sig_vy[0] = _read_P(kf)

    for i in range(1, n_conv):
        kf.predict(DT)
        kf.update(ObjectObservation(
            delta_x=noisy_obs[i, 0],
            delta_y=noisy_obs[i, 1],
            delta_theta=noisy_obs[i, 2],
            timestamp=i * DT,
        ))
        sig_x[i], sig_y[i], sig_vx[i], sig_vy[i] = _read_P(kf)

    # ── 找收敛帧：位置 σ < 1.5 × 稳态值（稳态取最后 50 帧均值）──────
    steady_x = float(np.mean(sig_x[100:]))
    steady_y = float(np.mean(sig_y[100:]))
    thresh_x = steady_x * 1.5
    thresh_y = steady_y * 1.5
    conv_frame = next(
        (i for i in range(5, n_conv)
         if sig_x[i] < thresh_x and sig_y[i] < thresh_y),
        50,
    )

    # ── 画图 ───────────────────────────────────────────────────────────
    C_X  = "#1565C0"   # 蓝
    C_Y  = "#AD1457"   # 品红
    C_VX = "#2E7D32"   # 深绿
    C_VY = "#E65100"   # 橙

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    # ── 上图：位置不确定度 ─────────────────────────────────────────────
    ax1.plot(frames, sig_x, color=C_X,  lw=2.0, label="σ_x (position X)")
    ax1.plot(frames, sig_y, color=C_Y,  lw=2.0, label="σ_y (position Y)", ls="--")
    ax1.axhline(SIGMA_POS * 1e3, color="#546E7A", ls=":", lw=1.5,
                label=f"Sensor noise floor  ({SIGMA_POS*1e3:.0f} mm)")
    ax1.axvline(conv_frame, color="#90A4AE", ls="--", lw=1.5, zorder=0)
    ax1.text(conv_frame + 2, ax1.get_ylim()[1] * 0.90,
             f"Converged\n(frame {conv_frame}, {conv_frame/FPS:.1f} s)",
             fontsize=9, color="#546E7A", va="top")
    ax1.set_ylabel("Position σ (mm)")
    ax1.set_title(
        f"Kalman Filter Cold-Start Convergence  "
        f"(Circular trajectory, 30 Hz,  σ_pos = {SIGMA_POS*1e3:.0f} mm)",
        fontsize=12,
    )
    ax1.legend(loc="upper right", framealpha=0.85)
    ax1.set_ylim(bottom=0)

    # 稳态标注（无箭头，直接标在曲线旁）
    ax1.text(
        n_conv * 0.65, steady_x + 0.3,
        f"Steady state ≈ {steady_x:.1f} mm",
        fontsize=9, color=C_X, ha="left", va="bottom",
    )

    # ── 下图：速度不确定度 ─────────────────────────────────────────────
    ax2.plot(frames, sig_vx, color=C_VX, lw=2.0, label="σ_vx (velocity X)")
    ax2.plot(frames, sig_vy, color=C_VY, lw=2.0, label="σ_vy (velocity Y)", ls="--")
    ax2.axvline(conv_frame, color="#90A4AE", ls="--", lw=1.5, zorder=0)

    # 速度稳态
    steady_vx = float(np.mean(sig_vx[100:]))
    ax2.axhline(steady_vx, color="#546E7A", ls=":", lw=1.5,
                label=f"Velocity steady state ≈ {steady_vx:.0f} mm/s")

    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Velocity σ (mm/s)")
    ax2.legend(loc="upper right", framealpha=0.85)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0, n_conv - 1)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Fig 3] {out_path.name}  saved")


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("MT3 Dynamic Alignment — Synthetic Validation")
    print(f"Parameters: {FPS} Hz, {DURATION:.0f} s, "
          f"v = {SPEED*100:.0f} cm/s, σ_pos = {SIGMA_POS*1e3:.0f} mm, "
          f"τ = {TAU*1000:.0f} ms")
    print(f"Output directory: {OUT_DIR}\n")

    _setup_style()

    plot_tracking_comparison(OUT_DIR / "tracking_comparison.png")
    plot_latency_compensation(OUT_DIR / "latency_compensation.png")
    plot_convergence(OUT_DIR / "convergence.png")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
