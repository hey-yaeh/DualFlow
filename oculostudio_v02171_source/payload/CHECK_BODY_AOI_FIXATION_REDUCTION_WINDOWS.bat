@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo OculoStudio is not installed in this payload folder.
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" tools\check_body_fixation_review_reduction.py --output docs\VALIDATION_v0_21_54_BODY_REVIEW_REDUCTION.json
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo Body AOI fixation review reduction validation FAILED.
  exit /b %EXIT_CODE%
)
echo Body AOI fixation review reduction validation PASSED.
endlocal
