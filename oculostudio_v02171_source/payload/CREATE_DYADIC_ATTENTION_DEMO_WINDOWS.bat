@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "DEMO=%CD%\OculoStudio_Dyadic_Attention_Demo"
if exist "%DEMO%" rmdir /s /q "%DEMO%"
"%OCULOSTUDIO_PY%" "%~dp0tools\create_dyadic_attention_demo.py" "%DEMO%" --overwrite
if errorlevel 1 (
  echo Demo creation failed.
  pause
  exit /b 1
)
"%OCULOSTUDIO_PY%" -m oculostudio studio --project "%DEMO%"
