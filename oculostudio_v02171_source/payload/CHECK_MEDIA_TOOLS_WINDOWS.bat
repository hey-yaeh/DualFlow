@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" -m oculostudio media-tools --test-privacy --require-ffprobe
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
