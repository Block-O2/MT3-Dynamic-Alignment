"""
测试辅助函数（可直接 import，不依赖 pytest）。

conftest.py 中的 fixture 和各 test_*.py 均从此处导入
make_object_cloud / make_circular_trajectory。
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def make_object_cloud(
    cx:        float,
    cy:        float,
    theta:     float,
    half_x:    float = 0.10,
    half_y:    float = 0.05,
    height:    float = 0.03,
    n_points:  int   = 300,
    noise_std: float = 0.003,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    在 (cx, cy)、朝向 theta 处生成长方体物体的合成点云。

    Returns
    -------
    cloud : shape (n_points, 3)，单位 m
    """
    if rng is None:
        rng = np.random.default_rng(42)

    lx = rng.uniform(-half_x, half_x, n_points)
    ly = rng.uniform(-half_y, half_y, n_points)
    lz = rng.uniform(0.0, height, n_points)

    c, s = np.cos(theta), np.sin(theta)
    wx = cx + c * lx - s * ly
    wy = cy + s * lx + c * ly

    return np.column_stack([
        wx + rng.normal(0, noise_std, n_points),
        wy + rng.normal(0, noise_std, n_points),
        lz + rng.normal(0, noise_std, n_points),
    ])


def make_circular_trajectory(
    R:        float = 0.30,
    omega:    float = 0.20,
    phi0:     float = 0.0,
    cx:       float = 0.5,
    cy:       float = 0.3,
    dt:       float = 1.0 / 30.0,
    n_frames: int   = 90,
) -> tuple[list, list, list]:
    """
    生成圆周运动合成点云序列。

    Returns
    -------
    clouds     : list of (N,3) 点云
    timestamps : list of float (s)
    gt_poses   : list of (x, y, theta)  真实位姿
    """
    rng    = np.random.default_rng(0)
    clouds, timestamps, gt_poses = [], [], []

    for i in range(n_frames):
        t     = i * dt
        x     = cx + R * np.cos(omega * t + phi0)
        y     = cy + R * np.sin(omega * t + phi0)
        theta = omega * t + phi0 + np.pi / 2

        clouds.append(make_object_cloud(x, y, theta, rng=rng))
        timestamps.append(t)
        gt_poses.append((x, y, theta))

    return clouds, timestamps, gt_poses
