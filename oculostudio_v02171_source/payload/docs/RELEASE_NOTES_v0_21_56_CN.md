# OculoStudio v0.21.56 — 时钟同步精度审计

## 为什么需要这次修复

v0.21.55 在手动进行同起点剪裁后，可能显示 `0.0 ppm` drift 和 `0.0 ms` P95 residual。同起点剪裁确实能让 offset 更容易恢复，但不能证明两个设备整段录制都没有时钟漂移。旧实现有两个问题让结果显得比证据更确定：

1. 每个窗口的 local lag 被限制在比较采样网格上；
2. 同一批窗口既用于拟合 affine map，又用于汇总残差。

## 同步修复

- **亚采样 local offset：**在相关峰附近做二次曲线精化，并保留峰宽、曲率和 lag uncertainty。
- **独立验证：**每个接受窗口依次留出，使用其他窗口拟合 affine map，再预测该留出窗口的 local offset；P95 使用 leave-one-window-out residual。
- **有限 drift 不确定度：**显示 drift 的 95% 置信区间、估计分辨率和支撑该估计的时间跨度。
- **不再制造“严格零”：**低于分辨率的残差标为 `precision-limited`，不能被解释为数学上的 0。
- **co-start 只是先验：**界面会提示手动剪裁使相对起点接近，但 drift 仍由整段独立窗口估计。
- **同步证据预览：**Calibration 可显示 Pupil IMU 与指定 ETM C3D 的对齐曲线、local offset 趋势、拟合 drift、接受/拒绝窗口、lag uncertainty 和交叉验证残差；不读取 ego video。
- **旧 lineage fail closed：**v0.21.55 的同步 artifact 可查看但不再 production-ready，必须重新运行以创建 v0.21.56 TimeGraph。

## 兼容性

项目和旧 artifact 都会保留，硬件 profile 与 wearer assignment 不变。接受新的 TimeGraph 后，需要重新生成 Gaze 3D 和后续 AOI，使其 lineage 引用新的精度审计映射。
