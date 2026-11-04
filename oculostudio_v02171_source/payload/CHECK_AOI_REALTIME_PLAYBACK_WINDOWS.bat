@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo  OculoStudio v0.21.54 real-time AOI playback validation
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: OculoStudio runtime not found. Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\check_aoi_review_realtime_playback.py --output-dir docs
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
