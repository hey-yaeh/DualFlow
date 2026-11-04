# OculoStudio v0.21.61 — 通过共享音频对齐 Facial-expression 与 Ego Video

## 功能目的

本版增加一套可审核的自动同步方法，用于：**facial-expression video 与 wearer ego / Pupil scene video 在同一场真实互动中由两个设备同时录制**的情况。

它对齐的是同一次真实声学事件，不会因为两段独立录制的视频说了相似句子，就把它们当成物理同步数据。

## 使用入口

打开：

> **Calibration → Paired summary**

或进入对应的：

> **R01 details / R02 details**

在右侧找到：

> **Shared-audio facial ↔ ego video sync · optional**

选择：

- wearer / interaction role；
- facial-expression video（source clock）；
- ego video 或 Pupil Native recording（reference clock）；
- 最大起始 offset 搜索范围。

点击：

> **Auto-align videos from shared audio**

任务只生成待审核 candidate，**不会自动改写 TimeGraph**。

## 同步模型

系统只拟合一个有物理意义的 affine 时钟关系：

```text
t_ego = scale × t_facial + offset
```

其中：

- `offset` 表示两个设备开始录制的时间差；
- `scale` / `drift ppm` 表示长时间录制中两台设备的时钟漂移。

处理流程包括：

1. 从两个视频中读取音轨，但不修改源视频；
2. 构建对麦克风增益、频响和噪声更稳健的能量/起始变化特征；
3. 粗略搜索全局 offset；
4. 在整段录制的多个时间位置进行局部窗口匹配；
5. 拒绝低一致性、峰值歧义、静音或不符合统一 affine 关系的窗口；
6. 某些只出现在一边的声音指令可以被当作局部异常窗口剔除；
7. 对局部 lag 做亚采样精化；
8. 鲁棒估计 offset 与 drift；
9. 用 leave-one-window-out 检查每个匹配窗口是否可由其他窗口预测。

系统**不使用任意 Dynamic Time Warping**，不会通过连续拉伸时间轴来制造表面一致性。

## 接受前必须审核视频

候选结果会打开双视频同步审核窗口，显示：

- offset；
- drift ppm；
- 接受窗口数与时间覆盖；
- leave-one-window-out P95 residual；
- 两条对齐后的音频 pattern；
- facial video 与 ego video 双播放器；
- **Review start / Review middle / Review end**；
- 音轨监听选择；
- 可选的固定 visual A/V correction；
- 必须勾选的视觉审核确认。

没有完成视觉确认时，后端会拒绝把候选发布为 TimeGraph。若设备存在稳定的内部音画延迟，可以输入固定修正量，该数值会进入正式审计记录，而不是被隐藏。

## 与 Pupil Native 的连接

当 reference 是 Pupil Native recording 时，接受后会发布两条连接：

1. facial-video MP4 PTS → Native ego-video MP4 PTS；
2. Native ego-video MP4 PTS → Pupil capture clock。

因此 facial-expression video 可以进入与 TSD、Gaze 3D 和 Vicon 相同的 TimeGraph。

## 安全门与可追溯性

- 解码前核验 interaction 与 R01/R02 role，避免跨互动或跨角色误配；
- candidate 阶段不会写入 TimeGraph；
- FAILED 或无关音频不会生成映射；
- 接受前重新检查源视频/NRD ZIP 是否被改动；
- 接受后生成不可变 Time Sync artifact；
- 发布过程是事务式的：若 artifact 或 Pupil clock bridge 创建失败，会回滚新映射并恢复旧映射；
- 不修改原始视频或 ZIP；
- accepted/rejected windows、相关指标、holdout residual、人工修正和审核确认均可追溯。

## 关于两边声音指令不同

短暂的单边指令可以被识别为不匹配窗口并剔除。系统只有在互动中仍存在足够多、分布足够广、共同支持同一 offset/drift 的声音证据时才会接受。

若共同音频太少、过度重复、只集中在一个很短片段，或者不同时间段不能由同一个 affine 映射解释，系统会标记 REVIEW 或 FAILED，而不是强行对齐。

## 限制

音频同步首先建立两条媒体时钟之间的关系。最终画面同步还可能受到设备内部 audio↔video latency 与少量声音传播延迟影响，所以开头、中间、结尾的视频审核必须保留。

若录制过程中发生暂停、重启、大段掉失或真正的时钟不连续，未来可能需要 piecewise-affine 或手动 anchor；v0.21.61 会 fail closed，不会进行无约束时间扭曲。

## 兼容性

本版不改变现有：

- R01/R02 wearer assignment 与 ETM 映射；
- Pupil IMU↔Vicon TimeGraph；
- Calibration handoff；
- C3D-defined aligned clips；
- aligned ego/TSD/FaceMapper package；
- Privacy、Gaze 3D、Face/Body AOI、Dyadic Attention；
- 人工审核决定。

只有需要把 facial-expression video 加入 TimeGraph 时，才需要运行新功能；原有数据无需重跑。
