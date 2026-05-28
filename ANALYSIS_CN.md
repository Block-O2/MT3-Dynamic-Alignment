# PyBullet 仿真分析

## 引言

本文总结 MT3 动态对准的 PyBullet 仿真研究：通过跟踪物体运动并在线变换示教轨迹，将静态物体上的 MT3 示教 replay 到缓慢移动的桌面物体上。主要结果来自 `ablation_20260524_233332`；`ablation_20260528_232432` 仅作为遮挡检查。

## 实验设置

共进行了两组 ablation。

| 运行 | 目录 | 设置 |
|-----|------|------|
| Run 1 | `ablation_20260524_233332` | 5 个条件，无遮挡，`SEED=42`，`N_TRIALS=20`，`TAU=0.1`，速度 `[2.0, 4.0, 6.0, 8.0]` cm/s |
| Run 2 | `ablation_20260528_232432` | 同样设置，额外加入 `OCCLUSION_PROB=0.15`；曾加入 `pd_feedforward`，但因实现 bug 失败，因此排除 |

每个条件-速度组合包含 20 次试验。所有条件都有较高方差，约为 +/-40-50% 标准差，因此结论应视为方向性结果。

## 结果

Run 1 成功率是主要结果。

| condition | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0 | 0 | 0 | 0 |
| raw_observation | 0 | 0 | 0 | 0 |
| dynamic_tau0 | 50 | 75 | 85 | 20 |
| uncertainty_gated | 45 | 60 | 45 | 35 |
| oracle_pose | 100 | 100 | 100 | 100 |

Run 2 在 15% 随机逐帧遮挡下的成功率如下。

| condition | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0 | 0 | 0 | 0 |
| raw_observation | 0 | 0 | 0 | 0 |
| dynamic_tau0 | 70 | 80 | 85 | 25 |
| uncertainty_gated | 40 | 60 | 50 | 30 |
| oracle_pose | 100 | 100 | 100 | 100 |

在 Run 1 的 `dynamic_tau0` 中，2 cm/s 的主导失败模式是 `attempt_limit`：机器人无法足够接近目标。8 cm/s 时 `lift` 占主导：夹爪闭合但错过物体。4-6 cm/s 的失败分布在多个类别中，更接近随机感知噪声。

## 关键发现

### 1. Static Replay 完全失败

`static_replay` 在所有速度下成功率均为 0%。直接在移动物体上 replay 静态示教不可行，因此动态适应是必要的。

### 2. Kalman 滤波是必要的

`raw_observation` 成功率同样为 0%。它与 `static_replay` 成功率相同，但失败原因不同：夹爪会闭合但错过物体，表现为 lift failure，而不是 `attempt_limit`。原始质心观测不够稳定。

### 3. 控制器不是瓶颈

`oracle_pose` 在所有速度下达到 100% 成功率。相同控制器在完美物体位姿下可以成功，说明限制因素是感知与状态估计。

### 4. 失败模式与速度相关

2 cm/s 时速度信号弱，Kalman 预测的信噪比较低。8 cm/s 时失败转向 lift miss，与控制带宽限制一致。Run 1 中最佳速度范围是 4-6 cm/s。

### 5. uncertainty_gated 在仿真中无法区分

`uncertainty_gated` 在 PyBullet 中与 `dynamic_tau0` 难以区分。Kalman 协方差 `P` 接近常量，因为噪声是固定且合成的，没有真实感知变化。这是仿真限制，不是方法缺陷。

### 6. 随机遮挡结构性不足

Run 2 表明，15% 随机逐帧遮挡没有产生足够结构化的感知变化，无法区分 `uncertainty_gated` 和 `dynamic_tau0`。下降过程中由机械臂造成的遮挡会更真实。

## 局限性

没有真实延迟时，无法验证 `tau` 补偿。当前模型只处理平面 SE(2)，不处理 6D 运动。实验只测试了抓取。每个条件 20 次试验导致高方差。PyBullet 接触动力学也不同于真实硬件。

## 需要的硬件验证

需要硬件实验来验证真实感知与执行延迟下的 `tau` 补偿，评估 `uncertainty_gated` 在运动模糊、遮挡和点云密度变化下的表现，并测试该方法是否能泛化到抓取以外的其他示教类型。
