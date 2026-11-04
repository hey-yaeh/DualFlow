@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_aoi_result_handoff.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Body AOI result-handoff checks passed.
) else (
  echo Body AOI result-handoff validation failed.
)
pause
exit /b %RESULT%
