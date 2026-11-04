@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
echo ============================================================
echo  OculoStudio v0.21.68 uninstaller
echo ============================================================
echo This removes the installed application and generated Python runtime.
echo Your recordings, project folders, exports, and annotations are NOT deleted.
echo.
choice /C YN /N /M "Continue uninstalling OculoStudio? [Y/N] "
if errorlevel 2 exit /b 0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\remove_shortcuts.ps1" >nul 2>nul
if exist "%OCULOSTUDIO_VENV%\" rmdir /s /q "%OCULOSTUDIO_VENV%"
set "APPDIR=%~dp0"
echo.
echo OculoStudio has been removed. User projects and recordings were preserved.
start "" cmd.exe /d /c "timeout /t 2 /nobreak ^>nul ^& rmdir /s /q ""%APPDIR%"""
exit /b 0
