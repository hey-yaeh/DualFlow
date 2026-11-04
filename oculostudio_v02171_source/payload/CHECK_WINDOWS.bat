@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo OculoStudio is not installed. Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" -m oculostudio check
pause
