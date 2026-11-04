# OculoStudio v0.21.60 — 成对 Calibration 视图与明确的模式下拉框

## 为什么需要这一版

此前有两个控件容易产生歧义：

- 右上角 Guided / Expert 模式按钮显示的是“当前状态”，用户难以判断点击后会进入哪个模式；
- Calibration 和其他 Expert 页面使用 `Use R01 / Use R02`，容易让人误以为只对一个角色进行 Calibration，或者点击会改变 wearer assignment。

现在 Calibration 已经是 R01 + R02 成对运行、成对验收的流程，因此原来的表达不再合适。

## 顶部模式改为下拉选择

右上角现在明确列出：

- **Guided workflow**
- **Expert tools**

下拉框中选中的值就是将要显示的模式，不再需要根据“当前模式按钮”猜测点击目标。模式选择仍会在本机保留。

## Calibration 改为成对总览

Calibration 默认进入 **Paired summary**，在同一个权威流程中同时显示两套角色证据：

- R01 → ETM-B
- R02 → ETM-A

Calibration 提供三个显示视图：

- **Paired summary**
- **R01 details**
- **R02 details**

R01/R02 details 只是只读的证据与 profile 详情视图。切换详情不会：

- 改变 wearer assignment；
- 交换 ETM-A 与 ETM-B；
- 改写角色同步结果；
- 替换 calibration profile；
- 重新生成 paired handoff；
- 改变其他页面当前处理的角色。

需要同时作用于两名角色的动作仍集中在 Paired summary，例如 wearer resolution、双角色时钟同步、handoff 创建和成对 C3D-defined clip 生成。

## 其他按角色处理的 Expert 页面

Trim、Privacy、Gaze 3D、Targets 等输出本身仍然按 wearer role 分开，因此继续保留角色选择。但原来的 `Use R01 / Use R02` 改为明确的 **Active role** 下拉框。它只决定当前查看、处理或导出的角色。

## 兼容性

本版保留：

- 原始录制与项目 manifest；
- wearer assignment 与硬件 profile；
- 角色专属 TimeGraph 与 calibration handoff；
- C3D-defined aligned clips；
- aligned ego/TSD/FaceMapper package；
- Privacy 视频；
- gaze ray、Face AOI、Body AOI 和融合 target artifact；
- 人工审核草稿及不可变 reviewed artifact。

仅安装本版不需要重新运行已有科学处理步骤。

## 验证范围

本版使用真实 FastAPI project state 与 Chromium 渲染链路验证了模式下拉框、Calibration 默认 paired summary、只读角色详情、其他页面 Active role 选择、JavaScript 语法、Python 编译、wheel 完整性、package manifest 和最终 wheel 安装。

Windows WebView2 桌面壳与批处理安装器仍需在用户的 Windows 机器安装后做最终确认。
