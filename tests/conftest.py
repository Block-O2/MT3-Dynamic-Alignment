"""
pytest 共用 fixture 和合成数据生成工具。

所有测试均使用合成数据，不依赖任何硬件或 ROS。

合成场景
--------
物体（长方形，尺寸 0.2m × 0.1m）在水平面上沿圆周运动：
    x(t) = cx + R · cos(ω·t + φ₀)
    y(t) = cy + R · sin(ω·t + φ₀)
    θ(t) = ω·t + φ₀ + π/2          （朝向与速度方向对齐）

默认参数（设计笔记第五节"范围内"场景）
    R  = 0.30 m      圆轨道半径
    ω  = 0.20 rad/s  角速度（≈ 11°/s）
    v  = R·ω = 0.06 m/s  线速度（< 10 cm/s 设计上限）
    φ₀ = 0            初始相位
    dt = 1/30 s       30 Hz 相机帧率
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # MT3_Plus/ (for dynamic_alignment)
sys.path.insert(0, _HERE)                        # tests/     (for helpers)

import numpy as np
import pytest

from dynamic_alignment.types import DemoData
from helpers import make_object_cloud, make_circular_trajectory  # noqa: F401  (re-export)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def circular_trajectory():
    """90 帧圆周轨迹：R=0.3m, ω=0.2rad/s, 30Hz"""
    return make_circular_trajectory()


@pytest.fixture
def static_demo_data():
    """单帧 demo（alignment 阶段用），末端在坐标原点"""
    T_WE = np.eye(4)
    T_WE[0, 3] = 0.4   # 末端 X = 0.4m（模拟抓取位置）
    T_WE[1, 3] = 0.2   # 末端 Y = 0.2m
    T_WE[2, 3] = 0.3   # 末端 Z = 0.3m（上方）
    return DemoData(poses=[T_WE], timestamps=[0.0])


@pytest.fixture
def multi_frame_demo_data():
    """10 帧 demo 序列（interaction 阶段用），末端做直线下压动作"""
    poses      = []
    timestamps = []
    for i in range(10):
        t = i * 0.05   # 50ms 步长
        T = np.eye(4)
        T[0, 3] = 0.4
        T[1, 3] = 0.2
        T[2, 3] = 0.3 - 0.02 * i   # Z 方向缓慢下降
        poses.append(T)
        timestamps.append(t)
    return DemoData(poses=poses, timestamps=timestamps)
