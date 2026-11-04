\
@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed
if not exist "%OCULOSTUDIO_PY%" (
  echo ERROR: OculoStudio runtime is not installed.
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  goto :failed
)
set "PYTHONPATH="
"%OCULOSTUDIO_PY%" "%~dp0tools\check_calibration_sync_precision.py"
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo.
echo Calibration synchronization precision audit failed.
pause
exit /b 1
