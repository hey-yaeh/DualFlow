@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

echo ============================================================
echo  OculoStudio v0.21.54 - rebuild generated Windows runtime
echo ============================================================
echo This removes only generated Python environments.
echo Projects, recordings, exports, and annotations are not modified.
echo.
echo Removing:
echo   %OCULOSTUDIO_VENV%
if exist "%OCULOSTUDIO_VENV%\" rmdir /s /q "%OCULOSTUDIO_VENV%"
if exist "%OCULOSTUDIO_VENV%\" goto :failed

if exist "%~dp0.venv\" (
  echo Removing the obsolete project-local .venv that caused long-path failures...
  rmdir /s /q "%~dp0.venv"
  if exist "%~dp0.venv\" goto :failed
)

echo.
echo Generated environments were removed. Starting a clean installation...
call "%~dp0INSTALL_AND_RUN_WINDOWS.bat"
exit /b %ERRORLEVEL%

:failed
echo.
echo ERROR: The generated environment could not be removed.
echo Close OculoStudio, Python, and any terminal using this folder, then run this repair again.
pause
exit /b 1
