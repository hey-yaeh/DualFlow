@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

echo ============================================================
echo  OculoStudio v0.21.54 raw Vicon body marker validation
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_motionstudio_body_profile.py" --output "%~dp0docs\VALIDATION_v0_21_52_RAW_VICON_BODY_PROFILE.json"
if errorlevel 1 goto :failed

echo.
echo PASS: raw Vicon full-body 10-finger pelvis, torso, limbs and ray intersection validated.
pause
exit /b 0

:failed
echo.
echo FAILED: raw Vicon body marker validation did not pass.
pause
exit /b 1
