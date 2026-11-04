@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo OculoStudio is not installed. Run INSTALL_OCULOSTUDIO.bat first.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0tools"
"%OCULOSTUDIO_PY%" "%~dp0tools\check_aligned_clip_export.py"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
  echo.
  echo Aligned clip export validation FAILED.
  pause
  exit /b %RESULT%
)
echo.
echo Aligned clip export validation PASSED.
pause
exit /b 0
