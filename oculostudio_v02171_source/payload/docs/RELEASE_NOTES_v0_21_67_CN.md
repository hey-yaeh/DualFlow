# OculoStudio v0.21.68 — Project 级 Calibration Sets

## 解决的问题

此前虽然可以将 S004 中的 ETM-A/ETM-B profile 应用到 S005–S007，但原始 Calibration 录制仍挂在某一个 Session 下，其他 Session 通过“复制 profile”共享。这不符合你的真实采集结构：一套参与者观看 AprilTag 的 Calibration 文件由几个 Session 共用，而每个 Session 又包含多个独立 Interaction。

本版将 Calibration 升级为真正的 **Project-level、可版本化 Calibration Set**。

## 新的数据层级

```text
Project
├── Calibration Sets
│   └── CAL_S004_S007
│       ├── rev_001
│       ├── P01 / P02 AprilTag calibration sources
│       ├── ETM-A / ETM-B profiles
│       └── applies to S004, S005, S006, S007
└── Sessions
    ├── S004
    │   ├── I01
    │   ├── I02
    │   └── ...
    ├── S005
    ├── S006
    └── S007
```

Calibration 原始文件只登记一次。S004–S007 保存的是指向同一不可变源文件和 active revision 的可审计引用，不复制 raw bytes。

## 创建 Project

三步向导的 Shared resources 页面现在可以创建初始 Calibration Set：

- Calibration Set ID，例如 `CAL_S004_S007`；
- 显示名称；
- Purpose：综合、硬件几何或 known-target gaze calibration；
- Applies to all created Sessions。

新 Project 使用 folder layout v3：

```text
calibration_sets/CAL_S004_S007/
├── calibration_set.json
├── revisions/rev_001/
├── sources/
├── derived/
└── reports/
```

## Data 页面

Data 页面明确分成：

1. **Project Calibration Sets**：跨 Session 共用的 AprilTag Calibration、profile、revision 和适用范围；
2. **Current Interaction data**：当前 Session / Interaction 的 NRD、TSD、FM、Body/ETM C3D、Facial Video、Props 等。

Calibration source 不再伪装成当前 Interaction 数据，也不会因为导入 Calibration 文件而自动创建无依据的 Global-Time mapping。

## Calibration Set Manager

现在可以：

- 新建多个 Calibration Set；
- 将一个 Set 绑定到多个 Session；
- 创建并激活新 revision；
- 挂接已有 profile；
- 刷新各 Session 继承的 source/profile reference；
- 查看每个 Session 正在继承哪个 Set/revision；
- 打开 Set 的实际文件夹。

新 revision 默认继承上一版的 sources 和 profiles；旧 Interaction artifact 继续记录其原 revision，不会被静默改写。

## Profile 生成与导入

新估计或导入的 ETM-A/ETM-B profile 可以直接写入当前 Calibration Set revision。绑定的 Session 会得到带 lineage 的 role-appropriate profile artifact，但 Calibration 原始录制仍只保存一份。

## 仍然按 Interaction 独立处理的内容

共享 Calibration Set 不代表共享时间同步。每个 Interaction 仍独立完成：

- wearer / ETM assignment；
- Pupil IMU ↔ ETM C3D offset 与 drift；
- TimeGraph；
- C3D-defined Trim；
- Privacy；
- Gaze 3D；
- Face / Body / Object AOI；
- Dyadic Attention。

准确关系是：

```text
一个 Calibration Set → 多个 Sessions → 每个 Session 多个独立 Interactions
```

## 旧项目兼容

旧项目仍可直接打开。已有 shared-calibration metadata 可以迁移为 Project Calibration Set；原始录制不移动，已有 artifacts、annotations 和 exports 不会被覆盖。
