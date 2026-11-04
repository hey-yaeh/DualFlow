@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_aoi_torso_surfaces.py"
if errorlevel 1 goto :failed
echo.
echo Torso geometry and Body review validation passed.
pause
exit /b 0
:failed
echo.
echo Validation failed.
pause
exit /b 1
