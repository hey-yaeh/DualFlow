@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_edge_tolerance.py"
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
