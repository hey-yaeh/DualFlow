@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" -c "import oculostudio.motionstudio_bridge as b; print('PASS: MotionStudio bridge import')"
"%OCULOSTUDIO_PY%" -c "import torch, smplx; print('PASS: optional parametric mesh runtime')"
echo.
echo If the second check fails, run INSTALL_MOTIONSTUDIO_BRIDGE_WINDOWS.bat.
pause
