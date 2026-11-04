# OculoStudio v0.21.62 — Body / Object AOI 直接复用 Gaze Ray 的 C3D 时间

## 修复的问题

有效的 Body AOI 任务可能在约 3% 时失败：

```text
No time-map path connects 'time_…' to 'global'
```

新的 Calibration / Gaze 3D 流程本身已经正确使用经过验收的 Pupil→ETM affine TimeGraph，并在每一个 gaze sample 中保存了真正用于插值 ETM 姿态的 Vicon/C3D 时间：

```text
gaze3d_c3d_time_sec
```

旧 Body AOI 仍根据 ray artifact 的 `alignment_mode=timegraph` 标记，错误地要求：

```text
Pupil Timeseries → project global → Body C3D
```

对于由同一 Vicon interaction 生成的 C3D-defined clip，这条绕行路径既没有必要，也可能根本不存在。

## 新的时间路径

现在 matching Native-reference trim 与 Calibration handoff 下，Body AOI 直接使用：

```text
每个 gaze sample
  → Gaze 3D 已保存的 gaze3d_c3d_time_sec
  → 在相同 Vicon 时间查询 self / partner Body C3D
```

因此不再要求额外建立 `Timeseries → global` 路径。

输出中会分开记录：

- `alignment_mode`：Gaze 3D 当初如何生成 world-space ray；
- `body_time_alignment_mode`：Body AOI 实际如何查询身体几何；
- `global_timegraph_required`：是否真的使用了 project Global Time；
- `gaze_ray_c3d_clock_column`：下游复用的精确时间列。

## 安全门

只有满足以下条件才会启用 direct C3D clock：

- 选择了 matching Native-reference trim；
- gaze-ray CSV 中存在精确 C3D query time；
- self / partner Body C3D 与 Calibration handoff 或 C3D-defined trim 中登记的 role-resolved body 一致；
- 至少 90% 的有效 gaze query time 落在每个所选 C3D 的时间范围内。

若选错 interaction、Body C3D 或 Object C3D，系统仍会 fail closed，并给出明确的时间范围错误。没有嵌入 C3D clock 的旧数据仍使用原来的 verified TimeGraph fallback。

## Object AOI

Object AOI 同时修复。与同一次 Vicon interaction 录制的 prop C3D 可以直接复用 gaze-ray C3D time；不匹配的 prop recording 会被时间覆盖安全门拒绝。

## 界面与 QC

Body AOI 面板现在明确说明：对于 C3D-defined interaction，优先使用 Gaze 3D 中的精确 C3D 时间，不需要单独的 Timeseries→Global 路径。结果页面会显示 `DIRECT C3D CLOCK` 或 project TimeGraph。

## 兼容性

已有项目、Calibration handoff、TimeGraph、aligned clips、Gaze 3D rays、Face AOI 和人工审核结果都继续有效。只需重新运行失败的 Body AOI；如果现有 gaze-ray CSV 已包含 `gaze3d_c3d_time_sec`，不需要重新生成 Gaze 3D。
