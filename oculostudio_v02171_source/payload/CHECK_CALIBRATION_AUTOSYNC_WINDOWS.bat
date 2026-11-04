@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
if not exist "%OCULOSTUDIO_PY%" (
  echo ERROR: OculoStudio runtime is not installed.
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0tools;%PYTHONPATH%"
"%OCULOSTUDIO_PY%" "%~dp0tools\check_calibration_autosync.py"
if errorlevel 1 (
  echo.
  echo Calibration auto-sync validation FAILED.
  pause
  exit /b 1
)
echo.
echo Calibration auto-sync validation PASSED.
pause
exit /b 0
