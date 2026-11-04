@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "PY=%OCULOSTUDIO_PY%"
set "WHEEL=%~dp0dist\oculostudio-0.21.68-py3-none-any.whl"
echo Repairing the FFmpeg runtime in the path-safe OculoStudio environment...
echo No raw recordings or project data are modified.
"%PY%" -m pip install --only-binary=:all: "imageio-ffmpeg>=0.6,<0.7"
if errorlevel 1 goto :failed
if exist "%WHEEL%" (
  "%PY%" -m pip install --force-reinstall --no-deps "%WHEEL%"
  if errorlevel 1 goto :failed
)
echo.
echo Testing executable and actual H.264 / AAC encoding...
"%PY%" -m oculostudio media-tools --test-privacy --require-ffprobe --install-windows
if errorlevel 1 goto :failed
echo.
echo PASS: FFmpeg and ffprobe are ready. Reopen Studio and retry the failed Privacy job.
echo This does not resume an old job at 84 percent.
echo ffprobe frame reading has also passed. Retry your NRD preview.
pause
exit /b 0
:failed
echo.
echo Repair or encoder check FAILED. Review the error above.
echo If a saved path is invalid, open Privacy - Configure paths, clear it and save.
echo Offline users can select an existing working ffmpeg.exe in that dialog.
pause
exit /b 1
