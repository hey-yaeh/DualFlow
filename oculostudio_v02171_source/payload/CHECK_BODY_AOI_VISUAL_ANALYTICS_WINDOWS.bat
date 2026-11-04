@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_aoi_visual_analytics.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Body AOI visual analytics checks passed.
) else (
  echo Body AOI visual analytics validation failed.
)
pause
exit /b %RESULT%
