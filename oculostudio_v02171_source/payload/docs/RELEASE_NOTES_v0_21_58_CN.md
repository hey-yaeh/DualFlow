# OculoStudio v0.21.58 — 对齐片段 QC 预览

## 修复的问题

自动 C3D-defined Trim 页面已经显示 scope、segment length、FPS、人脸框、landmarks、初步 AOI、gaze 点和左右语义等全部参数，但没有真正执行渲染的按钮。自动片段本身已经正确且为 TIMEGRAPH VERIFIED；缺失的是 UI 到既有 preview engine 的连接。

## 新的审核流程

Calibration 创建 R01/R02 成对片段后：

1. 打开 **Trim**，用 **Use R01 / Use R02** 选择角色。
2. 在 **2 · Aligned clip QC preview** 中设置审核范围与叠加内容。
3. 点击 **Generate QC preview**；Job Queue 会显示 **Render aligned clip QC preview**。
4. 任务完成后点击 **Open preview**，或使用中央卡片中的 **Open aligned clip preview**。
5. 修改任何预览参数后可点击 **Regenerate QC preview**。

中央已验证片段卡片也新增 **Review aligned clip video**，可直接启动默认的开头＋中间＋结尾代表性审核。

## 科学与数据安全

- 预览直接使用自动 trim artifact 的精确 `preview_id` 以及已经验证的 NRD `.time` / MP4 PTS 区间。
- 不会再次要求输入 start/end，也不会根据 nominal FPS 重新估计边界。
- 直接读取已经配对的 TSD gaze 和 FaceMapper 几何。
- 只产生临时缓存视频：不会注册新的科学 artifact，不会替换 aligned clip，也不会重新编码原始 NRD 视频。
- 当 Calibration 无法创建自动片段时，原有 manual recovery trim 与 pre-trim preview 仍然保留。

## R01/R02 隔离

预览按照自动 `preview_id` 保存，而不是共用一个全局“最后预览”。切换到 R02 时不会显示 R01 的播放器；切回 R01 或重新打开项目时会恢复 R01 对应的预览，R02 同理。

## 验证

本版定向验证覆盖：

- Generate/Open 按钮可见且可执行；
- R01/R02 均实际生成 FFmpeg/H.264 overlay 视频；
- FaceMapper 与 gaze 映射；
- 两个角色拥有不同的 preview/overlay ID；
- 角色切换不串用预览；
- 新浏览器文档中能够恢复预览；
- 不创建新的不可变 artifact，也不重新编码源视频；
- JavaScript 语法、Python 编译与 wheel 完整性。

自动浏览器验证使用 live FastAPI backend 与项目内置 offline relay harness。Windows WebView 桌面壳仍需在用户的 Windows 机器上做最终确认。

## 安装

关闭 OculoStudio，完整解压安装包 ZIP，然后运行 `INSTALL_OCULOSTUDIO.bat`。现有项目和已生成 artifact 均会保留。
