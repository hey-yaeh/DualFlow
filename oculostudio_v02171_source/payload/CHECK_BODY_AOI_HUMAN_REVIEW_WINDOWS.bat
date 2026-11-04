@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo OculoStudio is not installed. Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
echo ============================================================
echo  OculoStudio v0.21.54 Body AOI human-review validation
echo ============================================================
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_aoi_manual_review.py"
if errorlevel 1 goto :failed
"%OCULOSTUDIO_PY%" "%~dp0tools\check_body_human_review_ui.py" --output "%~dp0docs\OculoStudio_v0_21_48_BodyHumanReview.png"
if errorlevel 1 goto :failed
echo.
echo Body AOI human-review validation passed.
pause
exit /b 0
:failed
echo.
echo Body AOI human-review validation failed.
pause
exit /b 1
