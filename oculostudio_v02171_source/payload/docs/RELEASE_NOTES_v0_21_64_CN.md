# OculoStudio v0.21.64 — 角色锁定的 Partner 投影

## 修复的问题

在 Active role 为 **R02** 时，Gaze Targets 页面可能残留或自动选中属于 R01 的 **ETM-B** calibration profile。这样生成的 participant association 可能没有任何有效、位于画面内的 partner-head projection，Face AOI Review 中便只看到 FaceMapper 几何，而看不到浅蓝色的 partner 身份判断框。

同时，旧界面可能让另一个 Native-reference trim 生成的 participant association 继续出现在 R02 Face AOI 下拉框中，从而出现：

```text
Participant association and Face AOI must use the same Native-reference trim
```

## 角色 profile 现在由 Paired Calibration 锁定

Participant association 现在以 paired Calibration handoff 为唯一权威来源：

- R01 固定使用绑定到 **ETM-B** 的 profile；
- R02 固定使用绑定到 **ETM-A** 的 profile；
- 切换 Active role 不会继承另一个 wearer 的 profile；
- 手工选择错误 profile 时，后端会明确拒绝；
- 手工 override 仅保留在折叠的 emergency recovery 区域。

Gaze Targets 右侧会先显示只读提示，例如：

```text
R02 → ETM-A → <paired handoff 绑定的 profile>
```

## Participant Association、Face AOI 与 Trim 严格同源

以下下拉框现在按照**精确 Native-reference Trim artifact ID**过滤，而不是只看文件名或 R01/R02 文本：

- Filtered FaceMapper；
- Participant association；
- Trimmed Privacy video。

Face AOI 只会列出与当前 role、当前 trim 完全匹配的 association。若通过旧状态或 API 提交了不匹配结果，错误消息会同时显示两个 trim ID，并要求重建或选择正确 association。

## 蓝色投影改为 fail-closed

在 `Partner only` 模式下，participant association 必须至少生成一个有效、位于画面内的 partner-head projection。若整段为 0：

- 任务直接失败；
- 不发布不可用 artifact；
- 错误中显示当前 role、ETM 以及建议检查的 profile、角色分配、trim 和头部 markers。

浅蓝色框仍然**只用于 partner 身份 gate**；绿色 FaceMapper 框仍然是 face/no-face 的唯一边界，也是 eyes/nose/mouth AOI 的几何来源。若某一帧 partner 真实离开画面或头部 pose 无效，该帧没有蓝框是正常行为。

## 升级后需要重跑什么？

已有 Calibration、TimeGraph、Trim、Privacy、Gaze 3D、Body AOI、Object AOI 和人工审核结果不会被修改。

安装后，对于截图中的 R02，只需：

1. Active role 选择 `R02`；
2. 在 Gaze Targets 第 0 步重新运行 `Project heads & assign faces`；
3. 确认只读卡显示 `R02 → ETM-A`；
4. 使用新 association 重新生成 R02 Face AOI；
5. 打开新 Face AOI Review，检查 partner 位于画面内的帧是否出现浅蓝框。

不需要重新运行 Calibration、Trim、Privacy、Gaze 3D 或 Body AOI。
