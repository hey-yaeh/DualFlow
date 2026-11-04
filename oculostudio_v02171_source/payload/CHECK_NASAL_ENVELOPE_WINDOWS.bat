@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 exit /b 1
"%OCULOSTUDIO_PY%" "%~dp0tools\check_nasal_envelope_direct_accept.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Nasal envelope and direct FaceMapper-hit acceptance checks passed.
) else (
  echo Nasal envelope validation failed.
)
pause
exit /b %RESULT%
