# PyBullet 仿真分析

本说明与 [英文分析](ANALYSIS.md) 使用相同的原始记录。README 和个人网站统一展示 **2026-05-28、15% 随机点云丢帧**的 `ablation_20260528_232432`；2026-05-24 的无遮挡运行作为独立历史实验保留。展示选择不代表遮挡提高性能，也不覆盖旧数据。

## 实验来源

| 运行 | 条件 | 原始行数 |
|---|---|---:|
| [2026-05-24](simulation/results/canonical/ablation_20260524_233332) | seed 42，每格 20 次，2/4/6/8 cm/s，`TAU=0.1`，无人工丢帧 | 400 |
| [2026-05-28](simulation/results/canonical/ablation_20260528_232432) | 相同 seed、次数、速度和 TAU，额外 `OCCLUSION_PROB=0.15` | 480 |

2026-11-04 从 CSV 的 `success` 列只读汇总，确认每个条件和速度组合恰有 trial 1–20，无缺失或重复。可运行 `python simulation/summarize_canonical.py` 重新核验，不写入原始数据。

## 当前展示：5 月 28 日，15% 丢帧

| 条件 | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| raw_observation | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| dynamic_tau0 | 14/20 (70%) | 16/20 (80%) | 17/20 (85%) | 5/20 (25%) |
| uncertainty_gated | 8/20 (40%) | 12/20 (60%) | 10/20 (50%) | 6/20 (30%) |
| oracle_pose | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |

[图表](simulation/results/canonical/ablation_20260528_232432/grasping_success_rate.png) · [Demo](simulation/results/canonical/ablation_20260528_232432/grasping_demo.gif) · [CSV](simulation/results/canonical/ablation_20260528_232432/raw_trials.csv) · [配置](simulation/results/canonical/ablation_20260528_232432/config.json)

原始 CSV 还保留 80 条 `pd_feedforward` 试验，各速度均为 0/20。此前分析记载该条件有实现 bug，故不纳入五条件比较；当前已提交代码没有此条件，无法从配置记录的提交独立恢复其实现或核验 bug 细节。数据与元信息均未删除。

## 历史对照：5 月 24 日，无人工丢帧

| 条件 | 2 cm/s | 4 cm/s | 6 cm/s | 8 cm/s |
|---|---:|---:|---:|---:|
| static_replay | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| raw_observation | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| dynamic_tau0 | 10/20 (50%) | 15/20 (75%) | 17/20 (85%) | 4/20 (20%) |
| uncertainty_gated | 9/20 (45%) | 12/20 (60%) | 9/20 (45%) | 7/20 (35%) |
| oracle_pose | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |

旧 [plot 图](simulation/results/plot/grasping_success_rate.png) 对应这组百分比；[开发日志](EXPERIMENTS.md) 中 30%/80%/80%/30% 则来自更早的每格 10 次开发试验，不应混用。

## 条件与成功标准

Franka Panda 在 PyBullet 内生成并记录静态抓取轨迹，随后对移动方块回放；不加载原 MT3 真机示教，不训练或重训策略。方块沿半径 0.15 m 的往返半圆平移；控制频率 30 Hz，每步八个物理子步，每次回放最多 25 秒、三次尝试。抓取实验关闭 PCA yaw 估计。

- `static_replay`：直接回放原轨迹，不做在线物体变换。
- `raw_observation`：质心平移，无 Kalman/预测；无效点云时冻结示教时钟。
- `dynamic_tau0`：Kalman、**实际 `tau=0.1s`**、基于横向误差的自适应回放；名字是历史标签，不代表零预测。
- `uncertainty_gated`：相同跟踪与预测，用位置协方差量代替横向误差输入门控。
- `oracle_pose`：使用真值平移，示教时钟正常推进。

15% 丢帧是在感知条件的观测调用中随机用空点云替换整帧，不等于真实机械臂遮挡或实测相机延迟。

成功须同时满足：最终抬升 >50 mm；连接时相对物体坐标系示教目标的水平误差 ≤15 mm；夹爪轴向误差 ≤20°；连接前 0.5 秒内最大接近误差 ≤20 mm。方块靠近并闭爪后用固定约束连接，评估的是几何对准与回放，不是摩擦接触、触觉或力控制抓取。

## 结论边界

- 动态回放在这些条件下优于静态和原始观测回放；但滤波、预测和回放时钟同时改变，不能单独归因于 Kalman 或延迟补偿。
- 两组实验中，uncertainty gate 在 2/4/6 cm/s 的成功率均低于 dynamic replay，8 cm/s 略高。不能称二者等效，也不能据此确认协方差门控普遍有效；CSV 未归档协方差时序。
- Oracle 每格 20/20 说明理想平移信息在简化场景下可成功，不能排除真实感知、延迟和接触下的控制器限制。
- 5 月 28 日 dynamic replay 在 2 cm/s 的失败为 5 次 attempt_limit、1 次 approach；8 cm/s 为 15 次 attempt_limit。5 月 24 日 8 cm/s 也有 14 次 attempt_limit。旧文“高速动态回放主要为 lift miss”的说法与原始标签不符。
- 归档图上的“弱速度信号”“最佳区间”“控制带宽限制”属于机制假设，不能只凭计数证明。
- 单 seed、每格 20 次的比例不确定性较大；二值结果约 40–50 个百分点的标准差不等于成功率估计的置信区间。不宣称显著性或丢帧带来的因果改善。
- 当前抓取仿真没有注入实测端到端延迟，也没有受控 tau 对照。合成示例和单元测试不能证明真机延迟补偿效果。
- 跟踪器限于平面 SE(2)，本抓取实验只测试平移和单一任务。Sawyer/RealSense 仍未集成验证，所有真机表现均待确认。

复现命令、实际验证范围和历史源码快照缺口见 [REPRODUCING.md](REPRODUCING.md)。
