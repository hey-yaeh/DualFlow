# OculoStudio v0.21.55 — Calibration 内置 Pupil IMU ↔ Vicon 自动同步

## 为什么需要这一版

“预剪切同起点”只能说明两种数据在开头大致对齐，不能证明 Pupil 与 Vicon 的设备时钟在整段录制中没有漂移。v0.21.55 把同步变成 Calibration 的正式、可审计阶段，而不再只是 Gaze 3D 中的隐藏回退。

## Calibration 四阶段

1. **确认 interaction wearer。** wearer assignment 保持不可变；当前 OCUMO 配置为 R01→ETM-B、R02→ETM-A。
2. **准备硬件 profile。** ETM-A、ETM-B 各需要一个 measured/imported/validated profile。
3. **自动同步时钟。** 分别比较每个角色的 Pupil `imu.csv` 与该角色被分配的 ETM wearer C3D 刚体运动。
4. **创建 Gaze 3D handoff。** 把 wearer assignment、两套硬件 profile 和两套通过验收的 affine TimeGraph 绑定为新的不可变 handoff。

## 同步不使用 ego video

估计器只读取：

- Pupil Timeseries 中 `imu.csv` 的时间戳和陀螺仪；
- 已分配 ETM wearer C3D/NPZ 的 marker 轨迹与 Vicon 时间戳。

不会解码或分析 ego-video 像素。ego video 仍用于后续 Face AOI、overlay 和人工审核。

## 同时估计 offset 与 drift

程序先构建不依赖坐标轴朝向的角运动 signature，搜索全局 offset，再在录制的多个分布式窗口中重复匹配，最终拟合：

```text
t_vicon = a + b · t_pupil
```

界面和 artifact 会报告 offset、drift ppm、相关性、峰值分离度、有效窗口、median/P95 residual、有效重叠区间，以及精确的角色、输入资产和 wearer-assignment revision。

## 验收策略

- **HIGH：** 可在开启自动保存时直接写入 TimeGraph。
- **REVIEW：** 绝不静默晋级，必须由用户明确接受。
- **FAILED：** 不创建 TimeMap，也不改变 Calibration 状态。

只有 R01、R02 都具备已保存映射、正式接受、至少两个分布式 anchor/window，并且 provenance 明确没有使用 ego video，最终 handoff 才能进入 production-ready。

## Fail-closed 保护

- 明确标为另一角色的 Pupil Timeseries 会被拒绝；
- 不能复用属于旧 wearer-assignment revision 的同步结果；
- 指向错误 ETM 的 TimeMap 会被拒绝；
- 缺少任一角色同步时不能创建 handoff；
- 使用 video 或没有验证 drift 的同步不能满足新 Calibration contract；
- 旧的“仅 profile” handoff 不能再用于 Native-trim Gaze 3D；
- 一旦使用同步后的 handoff，Gaze 3D 禁止回退到 `pretrimmed_shared_start`。

## 升级后的推荐顺序

1. 打开 **Calibration**；
2. 确认 wearer assignment；
3. 确认 ETM-A、ETM-B profile；
4. 在第 3 阶段选择 R01、R02 对应的 Pupil Timeseries，运行双角色自动同步；
5. 对 REVIEW 候选核对 offset/drift 证据；
6. 在第 4 阶段创建同步后的 Gaze 3D handoff；
7. 若 TimeGraph 改变，重新生成 Gaze 3D 及后续 Body AOI。

旧项目、旧 artifact、人工审核决定和原始数据都不会被覆盖。
