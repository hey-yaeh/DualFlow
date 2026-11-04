# OculoStudio v0.21.70 — Data 页面双范围工作区与职责分离

## 本版解决的问题

此前 Data 页面同时存在 Calibration Set 卡片、一个语义不明确的全局拖放区、Interaction 导入按钮，以及跳转到 Calibration 页面继续管理源文件的入口。用户难以判断文件会进入共享 Calibration Set，还是当前 Session/Interaction；向已有 Calibration Set 追加文件也不直观。

v0.21.70 将源数据管理统一收口到 **Data**，并将 **Calibration** 收缩为纯计算流程。

## Data 页面：两个独立且完整的工作区

### Calibration Data

Project-level 共享数据区，包含：

- 当前 Calibration Set、revision 与适用 Sessions；
- 独立拖放区；
- 选择多个文件；
- 选择整个 Calibration 文件夹；
- 向现有 Set 追加文件；
- 管理 Set、revision 与 Session assignment；
- 已登记 Calibration sources 表格；
- Calibration source inspector。

支持 actor AprilTag NRD/TSD、calibration board/world C3D、ETM-A/ETM-B calibration C3D、board definition、`.oculocal` 和 `.g3dcal`。

浏览器拖放或选择的文件会分块复制到当前 Set 的 canonical `calibration_sets/CAL.../sources/` 中；桌面本地路径方式仍可只登记原始路径而不复制。

### Interaction Data

当前 Session/Interaction 的数据区，包含：

- 当前 `Session / Interaction` 上传目标；
- 独立拖放区；
- 选择多个 Interaction 文件或整个 Interaction 文件夹；
- identity / participant / role 编辑；
- 当前 Interaction 的资产表格与 inspector。

用于 NRD、TSD/TSDEV、FaceMapper、body/ETM C3D、facial video、props 和其他 interaction media。

表格只显示当前 I-code 与明确未分配的文件；同一 Session 中其他 Interaction 的文件被隐藏，避免误编辑。

## 全局拖放路由

- 拖到 Calibration Data 区：进入 Calibration Set 导入审核；
- 拖到 Interaction Data 区：进入当前 Interaction 导入审核；
- 拖到两个区之外：不猜测 scope，提示用户选择明确目标。

原来的无范围全局拖放区已删除。

## Calibration 页面：只做计算

Calibration 页面现在只负责：

1. Resolve R01/R02 wearer；
2. Build / bind ETM-A 与 ETM-B physical profiles；
3. Synchronize both Pupil clocks；
4. Validate and publish paired handoff；
5. Continue to Trim。

缺少源文件时，只显示 **Open Data → Calibration Data**。常规 Calibration 文件导入、追加、Set revision 和 Session assignment 不再放在 Calibration 页面。

高级 recovery tools 仍保留，以兼容遗留项目，但不再作为正常入口。

## 兼容性

本版不删除或重写：

- 已有 Calibration Sets / revisions；
- Session bindings；
- Interaction assets；
- hardware profiles；
- wearer assignments；
- clock TimeGraphs；
- Trim、Privacy、Gaze 3D、Face/Body/Object AOI；
-人工审核草稿与已发布 artifacts。

升级后无需重新导入或重新计算现有结果。

## 安装

1. 关闭 OculoStudio；
2. 完整解压安装 ZIP；
3. 运行 `INSTALL_OCULOSTUDIO.bat`；
4. 直接打开现有 Project。
