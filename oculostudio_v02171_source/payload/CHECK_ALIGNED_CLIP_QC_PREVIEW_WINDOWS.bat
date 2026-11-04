@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed
if not exist "%OCULOSTUDIO_PY%" (
  echo ERROR: OculoStudio is not installed. Run INSTALL_AND_RUN_WINDOWS.bat first.
  goto :failed
)
set "PYTHONPATH="
"%OCULOSTUDIO_PY%" "%~dp0tools\check_aligned_clip_qc_preview.py"
if errorlevel 1 goto :failed
echo.
echo Aligned clip QC preview validation passed.
pause
exit /b 0
:failed
echo.
echo Aligned clip QC preview validation failed.
pause
exit /b 1
