# OculoStudio v0.21.65 — 当前上下文锁定的 AOI 审核

## 修复的问题

当 Active role 为 R02、但当前 R02 trim 尚未生成 Face AOI 或 Body AOI 时，右侧 Human Review 与 AOI Review Player 可能回退到旧 R02 trim 或 R01 的 artifact，于是错误地提前显示审核按钮和待审数量。

## 新的上下文锁定规则

Gaze Targets 的所有界面现在统一按以下精确组合筛选证据：

- 当前 session；
- 当前 interaction；
- 当前 wearer role；
- 当前精确的 Native-reference trim artifact。

该规则同时约束顶部流程卡、统计区、右侧 Human Review、AOI Review Player、可选视频以及 Face + Body fusion 下拉框。当前上下文没有结果时保持为空，不再回退到其他角色或旧 trim。

## 审核入口的新行为

- 当前角色尚未生成 Face AOI 时，显示 `Generate R02 Face AOI first` 等明确占位提示。
- 尚未生成 Body AOI 时，显示 `Generate R02 Body AOI first`。
- 只有当前 role + 当前 trim 的 artifact 已存在时，才出现审核按钮。
- 待审数量只读取该 artifact。
- R01 或旧 R02 trim 的结果仍保留在 Results & QC 中，但不会冒充当前审核队列。
- 失败的生成任务不会产生审核入口。

## 兼容性

本版不改写任何科学结果。已有 Calibration、Trim、Privacy、Gaze 3D、Participant Association、Face AOI、Body AOI、Object AOI 及人工审核 artifacts 均保持不变，可继续在 Results 中查看。无需重跑此前步骤；只需为当前 R02 trim 正常生成新的 Face AOI / Body AOI。
