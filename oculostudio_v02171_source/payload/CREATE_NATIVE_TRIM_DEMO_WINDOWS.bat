@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
if exist "OculoStudio_NativeTrimDemo\project.oculostudio.json" goto :open
"%OCULOSTUDIO_PY%" -m oculostudio demo-native-trim "%CD%\OculoStudio_NativeTrimDemo"
if errorlevel 1 (
  echo Demo creation failed. Check ffprobe availability and the first error above.
  pause
  exit /b 1
)
:open
"%OCULOSTUDIO_PY%" -m oculostudio launch --project "%CD%\OculoStudio_NativeTrimDemo"
