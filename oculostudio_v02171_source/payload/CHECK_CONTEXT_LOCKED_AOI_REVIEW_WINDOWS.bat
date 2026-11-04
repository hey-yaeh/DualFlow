@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
"%OCULOSTUDIO_PY%" "%~dp0tools\check_context_locked_aoi_review.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Context-locked AOI review checks passed.
) else (
  echo Context-locked AOI review validation failed.
)
pause
exit /b %RESULT%
