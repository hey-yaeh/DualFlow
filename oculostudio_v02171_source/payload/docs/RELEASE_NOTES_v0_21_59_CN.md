# OculoStudio v0.21.59 — 对齐 Ego 视频、TSD 与 FaceMapper 正式导出

## 目的

自动 C3D-defined Trim 已经能够产生精确的角色专属 Native 时间区间，以及经过验证的 TSD 与 FaceMapper 子集，但此前没有发布可复用的干净对齐 ego-video。v0.21.59 在不改变既有对齐结果的前提下补上正式导出。

## 新的 Trim 导出

Calibration 创建并验证 R01/R02 clips 后，进入 **Trim**，选择当前角色，点击 **Export aligned ego + TSD + FM**。每次运行都会创建一个新的不可变 `aligned_clip_package` artifact。

可以任意组合导出：

- **干净的对齐 ego video**：不带 QC slate、人脸框、landmark、AOI 或 gaze overlay；
- **对齐后的 TSD ZIP**：从已验证 `TSD_trimmed.zip` 逐字节复制；
- **对齐后的 FaceMapper ZIP**：从已验证 `FaceMapper_trimmed.zip` 逐字节复制。

MP4 可以保留原始音频。支持 24、25、30、50、60 FPS。

## 时间契约

不可变 Native-reference trim 始终是唯一的边界依据。

1. 从父 trim manifest 读取 `start_media_ns` 与 `end_media_ns`；
2. 使用 trim 中记录的准确 scene-video member，重建原始 NRD `.time` / MP4 PTS 时钟；
3. 只允许选择父 `native_frame_map.csv` 中已验证的 Native frames；
4. 输出 `aligned_ego_frame_map.csv`，逐帧记录 CFR 输出帧对应的原始 Native frame index、展示区间、capture UTC 与选择误差；
5. 不要求用户重新输入，也不会重新推断开始或结束时间。

当输出 FPS 与 Native 实际帧节奏不同时，CFR 视频可能复制或省略源帧。后续分析必须使用显式 frame map，不能假设输出帧序号等于 Native frame 序号。

## Package 内容

完整组合导出包含：

```text
<role-specific aligned package>/
  <label>_ego.mp4
  aligned_ego_frame_map.csv
  <label>_TSD_aligned.zip
  <label>_FaceMapper_aligned.zip
  native_source_frame_map.csv
  source_trim_manifest.json
  aligned_clip_manifest.json
  aligned_clip_report.html
  README.txt
```

manifest 会记录父 artifact、角色、recording、精确区间、所选输出、视频参数、源校验和与 FFmpeg 命令。

## 安全性与兼容性

- 原始 NRD、TSD 与 FaceMapper 文件不会被修改；
- 父 Native-reference trim 不会被覆盖；
- TSD/FM 复制完成后进行 SHA-256 校验；
- Native 源发生变化或 recording ID 不匹配时 fail closed；
- R01/R02 使用独立父 trim、artifact 与输出目录；
- 支持仅视频或仅数据导出；
- 现有项目与历史 artifacts 继续有效；
- 临时 Aligned Clip QC Preview 继续保留，并与正式 package 明确分离。

## 验证摘要

- 合成 R01/R02 均成功创建独立不可变 artifact；
- 所有视频输出帧均位于父 trim 已验证 Native frame map 内；
- MP4 帧数与 `aligned_ego_frame_map.csv` 一致；
- TSD 与 FaceMapper SHA-256 与父 trim ZIP 完全一致；
- 不生成视频的 data-only package 可正常完成；
- API、artifact kind、中央 Trim 按钮、右侧输出控件与页面内正式视频播放器均已验证；
- 发布审计覆盖 Python 编译、JavaScript 语法、wheel RECORD、隔离环境安装与最终 ZIP 解压后复验。

## 安装

关闭 OculoStudio，完整解压安装包，然后运行：

```text
INSTALL_OCULOSTUDIO.bat
```

已有且已验证的 C3D-defined clips 无需重新导入、重新 Calibration 或重新 Trim。
