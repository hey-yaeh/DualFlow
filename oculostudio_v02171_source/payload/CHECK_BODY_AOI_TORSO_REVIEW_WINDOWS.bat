@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

echo ============================================================
echo  OculoStudio v0.21.54 torso + Body review validation
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_aoi_torso_review_reduction.py"
if errorlevel 1 goto :failed

echo.
echo Validation passed.
pause
exit /b 0

:failed
echo.
echo Validation failed. Review the first error above.
pause
exit /b 1
