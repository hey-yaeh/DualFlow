# OculoStudio v0.21.69 — Calibration Set 物理 Profile 构建器

## 本版解决的问题

在 v0.21.68 中，共享 Calibration 文件可以正确导入 `CAL_S004_S007`，但“从原始 AprilTag Calibration 录制生成 ETM-A / ETM-B profile”的处理链仍沿用旧的 Interaction role-detail 逻辑。这造成三类问题：

1. Profile estimator 按当前 Interaction wearer 查找 Calibration recording。当前 `S004/I02` 是 `R01=P01`、`R02=P02`，但共享 Calibration actors 是 `P01` 和 `P03`；P03 的录制因此不会被正确用于 ETM-A。
2. 旧入口要求 Calibration NRD、board C3D、tracker C3D 先存在 Project Global TimeGraph 路径；Project-level Calibration Set 导入又刻意不创建 Interaction TimeGraph，因此“文件已经上传”与“能够生成 profile”之间断开。
3. Profile 可能已经计算完成，却在写入 Calibration Set revision 或传播到 Sessions 时失败；页面只显示 `Profile missing`，容易诱导用户重复估计。

## 新的权威流程

物理 camera→tracker profile 现在由 Project-level Calibration Set 构建，而不是由当前 R01/R02 wearer 构建：

```text
CAL_S004_S007
├── P01 AprilTag calibration recording
├── P03 AprilTag calibration recording
├── calibration board C3D
├── ETM-A calibration tracker C3D
├── ETM-B calibration tracker C3D
└── world/reference C3D
        ↓
Automatic physical-ETM identity matching
        ↓
ETM-A camera→tracker profile
ETM-B camera→tracker profile
        ↓
One atomic Calibration Set revision
        ↓
S004 / S005 / S006 / S007 inherited references
```

Calibration actor 可以是 P01、P03；当前 Interaction wearer 可以是 P01、P02。物理 profile 绑定的是 ETM-A/ETM-B 硬件，不要求 actor 与当前 wearer 相同。

## UI 改动

### Data 页面

Calibration Set 卡片直接显示：

- 已登记 source 数量；
- ETM-A profile Ready/Missing；
- ETM-B profile Ready/Missing；
- `Build profiles` 主按钮。

### Calibration 页面

Phase 2 现在是：

> **Build physical ETM profiles from the shared Calibration Set**

R01/R02 detail views 仅用于查看证据与恢复，不再决定哪位 Calibration actor 属于哪个 ETM。

### 新的 Profile Builder

一个窗口中完成：

- source inventory；
- P01/P03 actor recording 检查；
- board、ETM-A、ETM-B、world C3D 检查；
- ETM-A/ETM-B 两个物理槽位；
- 自动匹配；
- 自动估计；
- 成对发布；
- 错误阶段与报告。

顶部和底部都提供同一个 `Analyze sources & build missing profiles` 操作，不需要滚动寻找旧的 role-specific estimator。

## 自动匹配与估计

系统对每个 Pupil Native AprilTag recording：

1. 检测 AprilTag board；
2. 使用 camera/board/tracker geometry 判断该 recording 对应 ETM-A 还是 ETM-B；
3. 使用 Calibration 专用的内部 offset search，而不要求 Project Global TimeGraph；
4. 鲁棒估计 camera→tracker SE(3)；
5. 验证输出硬件身份与预检一致；
6. ETM-A/ETM-B 均成功后，事务式写入同一 revision。

如果两段 actor recording 都匹配同一个 ETM、缺少某个 ETM、board 不是支持的格式、路径丢失或身份置信度不足，任务会 fail closed，不发布半套 profile。

## 失败恢复

如果旧任务已经生成了有效 profile artifact，但在 Calibration Set 注册阶段失败，新版会：

- 只查找显式标注为同一 Calibration Set、同一 revision、同一物理硬件的候选；
- 每个硬件只有一个安全候选时恢复绑定；
- 不采用任意 legacy profile；
- 不重复估计已经存在的 profile；
- 排除 `failed_uncommitted` 候选。

## 安装后怎么处理当前项目

1. 安装 v0.21.69；
2. 打开原 `OCUMO_S004_S007` Project；
3. 进入 **Data**；
4. 在 `CAL_S004_S007` 卡片点击 **Build profiles**；
5. 确认窗口列出 P01、P03、board、ETM-A、ETM-B、world 共 6 个 sources；
6. 点击 **Analyze sources & build missing profiles**；
7. 任务完成后，ETM-A 与 ETM-B 均应显示 Ready；
8. 返回 Calibration，已验证的 wearer assignment 与 R01/R02 clock sync 不需要重跑；
9. 创建/更新 paired handoff，然后继续 Trim/Gaze 3D。

若原始 source 自身存在 AprilTag、marker、C3D 或视频质量问题，新窗口会保留明确的失败阶段与错误原因；这类数据问题不会被静默改写。

## 保持不变

本版不修改：

- Calibration Set source bytes；
- wearer assignment 算法；
- Pupil IMU↔ETM/Vicon synchronization；
- C3D-defined Trim；
- aligned ego/TSD/FM；
- Privacy；
- Gaze 3D；
- Face/Body/Object AOI；
- facial-video shared-audio sync；
- 已有不可变 artifacts。
