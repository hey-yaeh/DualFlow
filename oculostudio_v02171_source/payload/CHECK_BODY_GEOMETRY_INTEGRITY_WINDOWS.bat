@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed
if not exist "%OCULOSTUDIO_PY%" (
  echo ERROR: OculoStudio runtime is not installed.
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_geometry_integrity_gate.py" --output "%~dp0docs\VALIDATION_v0_21_50.json"
if errorlevel 1 goto :failed
echo.
echo Body Geometry Integrity Gate validation passed.
pause
exit /b 0
:failed
echo.
echo Body Geometry Integrity Gate validation failed.
pause
exit /b 1
