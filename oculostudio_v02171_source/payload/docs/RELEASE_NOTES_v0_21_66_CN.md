# OculoStudio v0.21.66 — Project–Session 文件夹工作区

## 解决的问题

旧的新建项目窗口同时出现 parent folder、project folder、project name 和 first session，但没有清楚表达它们的层级；手工粘贴 parent path 还可能只改变文本而没有更新内部选择状态。用户也无法一次创建 S004–S007、在项目内统一管理 Session/Interaction，或明确复用项目级硬件 Calibration。

## 新的层级

```text
Project
├── Shared hardware Calibration
├── Session S004
│   ├── Interaction I01
│   ├── Interaction I02
│   └── ...
├── Session S005
├── Session S006
└── Session S007
```

R01/R02 是 Interaction 内的 role；P01/P02 等是匿名 participant，不与 Session 或 role 混用。

## 三步新建向导

### 1. Location

- **Parent directory**：容纳多个 OculoStudio Project 的上级目录，例如 `D:\DuoFlow\OculoStudio_Projects`。
- **New Project folder**：本项目实际文件夹，例如 `OCUMO_S004_S007`。
- **Project display name**：界面显示名称，例如 `OCUMO S004–S007`。
- 完整路径实时预览；手工输入、粘贴与文件夹选择器均会真实生效。
- 创建前检查 parent 是否存在/可写、目标是否为空，非空目标 fail closed。

### 2. Sessions

- 支持 `S004-S007`、`S004, S005, S006, S007` 或多行输入。
- 所有 Session 在一次事务中创建。
- 可选填写 Interaction template，例如 `I01-I06`；不同 Session 数量不同时可留空，之后逐个规划。

### 3. Shared resources

- 默认启用 Project-level Shared Calibration Library。
- 明确显示将创建的目录树和选定 Session。
- 创建完成后默认打开指定 Session。

## 新目录结构

新 Project 使用 folder layout v2：

```text
OCUMO_S004_S007/
├── project.oculostudio.json
├── shared/
│   └── calibration/
├── sessions/
│   ├── S004/
│   │   ├── session.oculostudio.json
│   │   ├── imports/
│   │   ├── derived/
│   │   ├── reports/
│   │   ├── annotations/
│   │   └── cache/
│   ├── S005/
│   ├── S006/
│   └── S007/
├── exports/
├── cache/
└── logs/
```

Browser/drag-and-drop 导入复制到当前 Session 的 `imports/`；桌面 picker 或粘贴本地路径仍可只保存原文件引用，不修改源数据。

## Project Session Manager

项目创建后可在 Home 中打开 **Manage Sessions & folders**：

- 查看全部 Session、Interaction/asset/result 数量及真实文件夹；
- 批量添加 `S008-S010`；
- 为任意 Session 规划 `I01-I06`；
- 切换当前 Session；
- 打开 Project、Session 或 Shared Calibration 文件夹；
- 不需要为每个 Session 另建 Project，也不需要手改 manifest。

## Shared Calibration Library

S004–S007 共用同一套 ETM-A/ETM-B 硬件几何 Calibration 时：

1. 在任一来源 Session（例如 S004）导入或估计 ETM-A/ETM-B profile；
2. 打开 Project Session Manager；
3. 选择来源 Session 和 profile；
4. 勾选 S005–S007，点击 **Apply selected profiles**。

系统会把 profile 的校验副本保存到 `shared/calibration/`，并为目标 Session 创建带 parent lineage、source hash 和 valid-session 范围的不可变 profile artifact。重复应用同一 profile 是幂等的，不会创建重复结果。

只共享 camera-to-tracker 硬件几何。以下内容仍必须按每个 Interaction 独立计算：

- wearer/ETM assignment；
- Pupil IMU ↔ ETM C3D offset 与 drift；
- TimeGraph；
- C3D-defined Trim；
- Privacy、Gaze 3D、Face/Body/Object AOI 与 Dyadic Attention。

## 兼容性

- 现有项目不会被移动或重写，继续使用其 legacy paths。
- 旧的人类可读标签如 `Session S001` 仍可打开和创建。
- 既有 Calibration、Trim、Privacy、Gaze 3D、AOI、Review 与 Results artifacts 均保持不变。
- 新目录结构只对新建的 folder-layout-v2 Project 自动启用。
