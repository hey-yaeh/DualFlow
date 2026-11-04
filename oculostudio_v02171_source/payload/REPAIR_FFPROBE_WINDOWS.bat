@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
echo ============================================================
echo OculoStudio - repair complete FFmpeg + ffprobe tools
echo Close OculoStudio first. No project or recording is modified.
echo ============================================================
if not exist "%OCULOSTUDIO_PY%" (
  echo ERROR: Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" "%~dp0tools\repair_media_runtime.py"
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" (
  echo Both tools passed. Reopen Studio and Preview the same clip again.
) else (
  echo Repair failed. Read the message above; do not rename ffmpeg.exe.
)
pause
exit /b %CODE%
