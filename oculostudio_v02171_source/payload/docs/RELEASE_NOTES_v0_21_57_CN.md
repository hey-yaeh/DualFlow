# OculoStudio v0.21.57 — C3D 定义的 Pupil 自动裁剪

## 目标

当 interaction 的 R01/R02 body C3D 已保留权威 Vicon 首尾时间时，正常流程不再要求手动输入 ego video 的开始和结束时间。

新版默认链路：

```text
原始 NRD + TSD + FaceMapper
        ↓
Calibration：wearer mapping + Pupil↔Vicon 精度审计 TimeGraph
        ↓
R01/R02 body C3D 的共同 Vicon 区间
        ↓ 反向映射各角色 TimeGraph
R01/R02 各自的 Pupil 时间区间
        ↓ 使用 NRD .time 与真实 MP4 PTS
成对 Native-reference trim
        ↓
Privacy → Gaze 3D → Face/Body AOI
```

## 核心更新

- 正常工作流改为 **Data → Calibration → Trim 检查 → Privacy → Gaze 3D**。
- 物理 interaction 区间取 R01/R02 body C3D 的半开共同交集。
- 分别反算 R01 与 R02 已接受的 affine TimeGraph，把同一 Vicon 区间映射到各自 Pupil 时钟。
- 使用 NRD 配套 `.time` 数组与真实 MP4 PTS 定位帧，不使用 `时间 × 标称 FPS`。
- TSD 与 FaceMapper 使用同一个角色级 Pupil 区间裁切。
- 两个角色采用事务式发布：任意一方失败时，不发布半套结果。
- 新增不可变 `aligned_clip_set`，把两个 role trim、共同 Vicon 区间和同步 Calibration handoff 绑定在一起。
- 原始 NRD MP4 不重写、不重编码。系统保存虚拟视频区间、精确 frame map 和裁切后的表格；后续 Privacy 可按同一 frame map 生成物理视频。
- 手动开始/结束时间仅保留在折叠的恢复入口中。

## 安全验收

以下情况会在发布前阻止：

- TSD 不是该角色已接受同步所使用的 Timeseries；
- R01/R02 body C3D 没有共同区间或重叠不足；
- 已分配 ETM C3D 无法覆盖共同区间；
- 反算后的 Pupil 区间超出 NRD 时钟容差；
- NRD、TSD、FaceMapper 不属于同一 Pupil recording；
- 任意一个角色导出失败。

每个结果都会记录共同 Vicon 区间、角色 Pupil 区间、NRD 媒体区间、帧边界误差、TimeGraph lineage、C3D 来源，以及“未手动输入时间码／未重编码源视频”的 provenance。

## 升级后流程

1. 安装 v0.21.57。
2. 完成 Calibration 的 wearer assignment、精度审计同步和 synchronized Gaze 3D handoff。
3. 在 Calibration 第 5 阶段检查自动选中的 R01/R02 NRD、同步 TSD 与 FaceMapper。
4. 点击 **Create C3D-defined R01 + R02 clips**。
5. 进入 **Trim** 检查共同物理时长、NRD 精确区间和边界误差。
6. 使用新 trim 继续 Privacy、Gaze 3D 与 AOI。

旧 trim 不会删除；新结果以带完整 TimeGraph lineage 的不可变 artifact 保存。
