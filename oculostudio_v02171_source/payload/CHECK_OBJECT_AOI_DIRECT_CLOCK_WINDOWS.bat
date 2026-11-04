@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo OculoStudio is not installed. Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "OUT=%~dp0docs\VALIDATION_v0_21_62_OBJECT_AOI_DIRECT_CLOCK.json"
"%OCULOSTUDIO_PY%" "%~dp0tools\check_object_aoi_direct_c3d_clock.py" > "%OUT%"
set "RC=%ERRORLEVEL%"
type "%OUT%"
if not "%RC%"=="0" (
  echo.
  echo Object AOI direct-clock validation FAILED.
  pause
  exit /b %RC%
)
echo.
echo Object AOI direct-clock validation PASSED.
pause
