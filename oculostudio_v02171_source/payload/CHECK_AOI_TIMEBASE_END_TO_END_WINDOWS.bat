@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

echo ============================================================
echo  OculoStudio v0.21.54 AOI timebase end-to-end validation
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_aoi_timebase_end_to_end.py" --output "%~dp0docs\VALIDATION_v0_21_52_AOI_TIMEBASE_END_TO_END.json"
if errorlevel 1 goto :failed

echo.
echo PASS: Body gaze-sample clock and Face track-aware gaze-time interpolation validated.
pause
exit /b 0

:failed
echo.
echo FAILED: AOI timebase end-to-end validation did not pass.
pause
exit /b 1
