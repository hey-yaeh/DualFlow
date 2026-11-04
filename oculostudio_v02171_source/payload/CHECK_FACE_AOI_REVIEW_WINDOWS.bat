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
echo  Checking strict green-face membership and environment fallback
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_nasal_envelope_direct_accept.py"
if errorlevel 1 goto :failed
"%OCULOSTUDIO_PY%" "%~dp0tools\check_strict_green_gate_environment.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking overlap-only identity gate and balanced lip AOI
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_balanced_facial_aoi.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking adaptive review-load reduction and legacy compatibility
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_aoi_review_load_reduction.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking expanded eyes, nose bridge, balanced lips and clip statistics
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_aoi_feature_geometry.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking temporal-consensus automatic review reduction
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_aoi_auto_review_reduction.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking Face AOI human review workflow
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_face_aoi_manual_review.py"
if errorlevel 1 goto :failed
echo.
echo ============================================================
echo  Checking local unblurred review and FaceMapper diagnostics
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_aoi_review_unblurred_facemapper.py" --no-browser
if errorlevel 1 goto :failed
echo.
echo All Face AOI checks passed.
pause
exit /b 0
:failed
echo.
echo Face AOI validation failed.
pause
exit /b 1
