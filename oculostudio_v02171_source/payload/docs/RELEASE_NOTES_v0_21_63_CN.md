# OculoStudio v0.21.63 — 对齐后的 Facial-Expression Video 正式导出

## 目的

本版补全 facial video 的正式工作流。Facial-expression video 通过 **Shared-audio facial ↔ ego video sync** 与对应角色的 ego/Pupil recording 同步并完成开头、中间、结尾视觉审核后，OculoStudio 现在可以把它导出到与 automatic C3D-defined clip 完全相同的物理互动区间。

## 新增 Trim 操作

进入 **Expert tools → Trim**，选择当前角色，然后使用：

> **4 · Export aligned facial video**

只有当前角色同时满足以下条件时按钮才会启用：

1. 已存在 automatic C3D-defined Native-reference Trim；
2. 已存在经过 start/middle/end 视觉审核并接受的 Shared-audio Time Sync artifact；
3. Shared-audio reference 与该 Trim 使用的是同一个原始 Native ego recording；
4. facial 与 ego 源媒体自同步后没有变化；
5. facial video 完整覆盖所需物理区间。

## Offset 与 clock drift 如何应用

已接受的映射是 affine TimeGraph：

```text
ego/reference time = scale × facial-source time + offset
```

系统不是只根据两个近似时间码简单裁切。它以 ego/reference clock 逐帧建立输出时间，使用已接受映射的逆变换查询原 facial video 的真实 frame PTS，并将完整 lineage 写入 `aligned_facial_frame_map.csv`。因此，起始 offset 与整段录制中的设备 clock drift 都会真正应用到导出视频。

勾选保留音频后，系统会提取对应的 facial audio 区间，按同一个 affine scale 的逆比例做轻微变速，再合并到干净的 MP4。系统不会使用任意 Dynamic Time Warping。

## 正式 artifact 内容

每个角色独立生成一个不可变的 `aligned_facial_video` artifact：

```text
<role>_aligned_facial_video.mp4
aligned_facial_frame_map.csv
aligned_facial_manifest.json
aligned_facial_sync_report.html
audio_video_sync_accepted.json
source_trim_manifest.json
native_source_frame_map.csv
README.txt
```

逐帧映射记录：

- output frame index 与 PTS；
- interaction/reference time；
- 对应 Native ego frame index 与 PTS；
- 查询的 facial-source time；
- 选中的原始 facial frame index 与 PTS；
- 源帧选择误差；
- 是否复用源帧、是否位于首尾边界。

## Lineage 与 fail-closed 安全门

正式 artifact 同时继承两个不可变 parent：

- 已接受的 role-specific Shared-audio Time Sync；
- role-specific automatic Native-reference Trim。

导出器不会重新估计同步，也不会重新计算 Trim 边界。角色/interaction 不匹配、ego reference 错误、缺少视觉确认、源文件变化、clock scale 无效、TimeGraph 缺失或 source coverage 不完整时都会拒绝发布。原始 FV、Native recording、TSD、FaceMapper、Calibration、Gaze 3D、AOI 和既有 artifacts 均不会被修改。

## Guided workflow

Interaction workspace 现在为每个角色显示一个可选 facial-video 分支：

```text
Shared-audio FV ↔ ego TimeGraph
→ Export drift-corrected aligned facial video
```

没有 facial-expression video 的 interaction 不受影响，仍可沿原有 ego/TSD/FM、Privacy、Gaze 3D、AOI 与 Dyadic 流程继续。

## 兼容性

既有项目和 artifacts 全部保留。升级后不需要重新运行 Calibration、Trim、Privacy、Gaze 3D、Face AOI、Body AOI、Object AOI 或已有的 shared-audio sync。已经接受并完成视觉审核的 Shared-audio Time Sync 可以直接用于正式 FV 导出。
