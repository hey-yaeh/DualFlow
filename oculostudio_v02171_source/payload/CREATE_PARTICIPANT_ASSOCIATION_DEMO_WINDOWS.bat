@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "OUT=%CD%\OculoStudio_Participant_Association_Demo"
if exist "%OUT%" rmdir /s /q "%OUT%"
"%OCULOSTUDIO_PY%" "%~dp0tools\create_participant_association_demo.py" "%OUT%" --overwrite || pause
"%OCULOSTUDIO_PY%" -m oculostudio --browser --project "%OUT%"
