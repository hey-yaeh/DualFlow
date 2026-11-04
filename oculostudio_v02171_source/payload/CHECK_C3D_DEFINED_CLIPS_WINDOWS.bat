@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo OculoStudio runtime not found. Install the application first.
  pause
  exit /b 1
)
"%PYTHON%" "%~dp0tools\check_c3d_defined_clips.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo C3D-defined paired clip validation passed.
) else (
  echo C3D-defined paired clip validation failed.
)
pause
exit /b %RESULT%
