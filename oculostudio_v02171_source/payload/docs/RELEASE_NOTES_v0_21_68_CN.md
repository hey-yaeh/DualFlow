# OculoStudio v0.21.68 — 直接 Calibration Set 导入工作区

## 解决的问题

v0.21.67 中，从 Data 页面点击 **Import Calibration files** 后会跳转到 Calibration 页面。真正的文件选择器位于右侧面板下方的折叠恢复区域，用户还需要滚动、展开并区分它与 Interaction wearer/sync 设置。Calibration Set 文件和 Interaction 文件因此在视觉上容易混淆。

v0.21.68 将两种导入流程彻底分开：

- **Project-level Calibration Set files**：跨多个 Session 共享的 AprilTag calibration recordings、Calibration C3D、board/world C3D、board definition 和 profile；
- **Current Interaction files**：当前 Session / Interaction 的 NRD、TSD、FM、body/ETM C3D、facial video 和 props。

## 新的 Data 页面

Data 页面顶部现在显示两个并列的 scope 卡片：

1. **Calibration Set files** — 打开专用 Calibration 导入窗口；
2. **Interaction recording files** — 打开当前 Interaction 的普通导入窗口。

原来的大号拖放区现在明确标记为 **Current Interaction only**，不会再暗示 Calibration recordings 也应拖到这里。

## 专用 Calibration Set 导入窗口

点击 **Import Calibration Set files** 后，应用保持在 Data 页面并直接打开大窗口。窗口在选文件前就显示：

- Calibration Set ID 与 display name；
- active revision；
- 适用的 Sessions；
- canonical project destination；
- Project-level scope 说明。

用户可以：

- 选择整个 Calibration 文件夹；
- 一次选择多个文件；
- 扫描而不修改项目；
- 在正式导入前检查每个文件的用途、participant/device、大小和 duplicate 状态；
- 一次提交为可审计的 batch job。

## 自动识别

扫描器可识别常见命名：

- `P01_calibration_*.zip` → actor / participant P01 known-target recording；
- `P03_calibration_*.zip` → actor / participant P03 known-target recording；
- `eye_tracker_A_*.c3d` → ETM-A tracker calibration motion；
- `eye_tracker_B_*.c3d` → ETM-B tracker calibration motion；
- `calibration_board_*.c3d` → AprilTag board motion；
- `markers_world_*.c3d` → world/reference-marker motion；
- board-definition JSON 与 `.oculocal` / `.g3dcal` profiles。

自动识别只生成待确认的导入计划。用户点击最终 Import 前，项目不会发生改变。

## 安全与兼容性

- Calibration Set sources 仍只登记一次，不为 S004–S007 复制多份 raw bytes；
- Session 继续保存 revision binding 与 lineage；
- Calibration source 不创建 Interaction TimeGraph；
- Interaction importer 保持独立；
- 旧 Calibration 页面中的 path scanner 仍保留为 **Advanced recovery only**；
- 现有 projects、Calibration Sets、Sessions、Interactions、artifacts 和 annotations 不需要迁移或重新生成。

## 推荐使用方式

对于示例文件夹：

```text
D:\DuoFlow\2_trimmed_data\S007\EyeCalibration
```

进入：

```text
Data
→ Calibration Set files
→ Import Calibration Set files
→ Choose calibration folder
→ Scan 6 files
→ Review detected roles
→ Import all 6 sources
```

不再需要进入 Calibration 页面，也不需要滚动寻找折叠入口。
