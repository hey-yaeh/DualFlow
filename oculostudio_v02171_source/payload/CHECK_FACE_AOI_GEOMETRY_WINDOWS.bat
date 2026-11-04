@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)

echo ============================================================
echo  Checking nasal bridge and both-alae envelope
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_nasal_envelope_direct_accept.py"
if errorlevel 1 goto :failed

"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_edge_tolerance.py"
if errorlevel 1 goto :failed

"%OCULOSTUDIO_PY%" "%~dp0tools\check_strict_green_gate_environment.py"
if errorlevel 1 goto :failed

"%OCULOSTUDIO_PY%" "%~dp0tools\check_balanced_facial_aoi.py"
if errorlevel 1 goto :failed

"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_aoi_feature_geometry.py"
if errorlevel 1 goto :failed

echo.
echo All Face AOI geometry checks passed.
pause
exit /b 0

:failed
echo.
echo Face AOI geometry validation failed.
pause
exit /b 1
