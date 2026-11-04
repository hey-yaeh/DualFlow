@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
set "DEMO=%~dp0OculoStudio_PrivacyAOIDemo"
if exist "%DEMO%\project.oculostudio.json" goto :open
"%OCULOSTUDIO_PY%" "%~dp0tools\create_privacy_aoi_demo.py" "%DEMO%" --overwrite
if errorlevel 1 (
  echo Privacy/AOI demo creation failed. Review the first error above.
  pause
  exit /b 1
)
:open
"%OCULOSTUDIO_PY%" -m oculostudio launch --project "%DEMO%"
