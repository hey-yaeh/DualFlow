# OculoStudio v0.21.71 — AprilTag Board Geometry 独立解析与 Profile Builder 修复

## v0.21.71

- 修复六个 Calibration source 已完整上传，但 Profile Builder 仍在 45% 报错 `The legacy engine did not produce a usable 4x4 board mapping` 的问题。
- 将两类独立证据彻底拆开：带标签的头部 marker 用于判断物理 ETM-A / ETM-B；scene AprilTag 与 16-marker 平面标定板用于解析 4×4 Tag 角点方向。
- 将确认后的 16 个 Tag 角点和 4 个外边界角保存到当前 Calibration Set revision，ETM-A、ETM-B 共用，并在后续重建时直接复用。
- 要求不同 Calibration actor 对同一个 board orientation 达成一致；若几何不完整或两段录制推断冲突，会在 Profile 估计前 fail closed。
- 增加 Calibration source preflight，明确显示 board marker 数量、C3D 帧数/频率、mapping 状态以及具体阻断原因，不再运行到 45% 后只显示模糊错误。
- 显式将 board mapping 传入原有实测 camera-to-tracker estimator；AprilTag detection、PnP、Calibration 内部时钟搜索、鲁棒 SE(3) 聚合和 QC 算法保持不变。
- Data 页面和 Project-level Profile Builder 同时显示 `Board mapped / pending`、ETM-A、ETM-B 三项 readiness。
- 已用你上传的 `CAL_S004_S007` 六个真实文件完成端到端验证：P01/James→ETM-B，P03/Jessica→ETM-A，两套完整 production profile 事务式发布成功。

# OculoStudio v0.21.71 — Data 页面双范围工作区

所有源数据管理现在统一放在 **Data**：

- **Calibration Data**：Project-level Calibration Sets、revision、共享源文件和 Session 绑定；
- **Interaction Data**：当前 Session/Interaction 的 NRD、TSD、FaceMapper、C3D、视频和 props。

两个范围各自拥有独立拖放区、选择器、文件清单与 inspector。Calibration 页面只进行校准计算。

完整说明见 `docs/RELEASE_NOTES_v0_21_70_CN.md`。
