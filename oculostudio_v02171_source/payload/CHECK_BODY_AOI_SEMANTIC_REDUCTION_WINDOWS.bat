@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo  OculoStudio v0.21.54 semantic Body review reduction check
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: OculoStudio runtime not found. Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\check_body_semantic_review_reduction.py --output docs\VALIDATION_v0_21_54_BODY_SEMANTIC_REDUCTION.json
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
