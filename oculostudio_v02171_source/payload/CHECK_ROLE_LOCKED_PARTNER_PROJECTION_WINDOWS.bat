@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
"%OCULOSTUDIO_PY%" "%~dp0tools\check_role_locked_partner_projection.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Role-locked partner projection checks passed.
) else (
  echo Role-locked partner projection validation failed.
)
pause
exit /b %RESULT%
