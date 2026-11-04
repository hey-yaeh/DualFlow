@echo off
setlocal
cd /d "%~dp0"
call tools\windows_runtime.bat
if errorlevel 1 exit /b %errorlevel%
"%OCULO_PYTHON%" tools\check_gaze_sample_clock_aoi.py --output docs\VALIDATION_v0_21_52_GAZE_SAMPLE_CLOCK_AOI.json
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" pause
exit /b %CODE%
