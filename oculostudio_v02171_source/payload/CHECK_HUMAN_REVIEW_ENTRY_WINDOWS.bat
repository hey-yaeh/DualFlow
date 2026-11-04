@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

set "PY=%OCULOSTUDIO_PY%"
if not exist "%PY%" goto :failed

echo ============================================================
echo  OculoStudio v0.21.54 Human Review entry validation
echo ============================================================
"%PY%" "%~dp0tools\check_human_review_entry.py" --output "%~dp0docs\OculoStudio_v0_21_48_HumanReviewEntry.png"
if errorlevel 1 goto :failed

echo.
echo Validation passed.
pause
exit /b 0

:failed
echo.
echo Validation failed. Review the first error above.
pause
exit /b 1
