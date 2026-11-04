@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if errorlevel 1 goto :failed

goto :main

:try_launcher
if defined PYTHON_EXE exit /b 0
py.exe -%~1 "%~dp0tools\windows_python_check.py" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_EXE=py.exe"
set "PYTHON_ARGS=-%~1"
exit /b 0

:try_python
if defined PYTHON_EXE exit /b 0
python.exe "%~dp0tools\windows_python_check.py" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_EXE=python.exe"
set "PYTHON_ARGS="
exit /b 0

:main
echo ============================================================
echo  OculoStudio v0.21.71 - Path-safe Windows setup
echo ============================================================
echo Application folder:
echo   %OCULOSTUDIO_APP_ROOT%
echo Runtime environment:
echo   %OCULOSTUDIO_VENV%
echo.

if exist "%~dp0.venv\" (
  echo NOTE: A legacy project-local .venv was found.
  echo It will be ignored because deep project paths can trigger WinError 206.
  echo The new short runtime location shown above is used instead.
  echo.
)

echo Locating a supported Python interpreter...
set "PYTHON_EXE="
set "PYTHON_ARGS="

where py.exe >nul 2>nul
if not errorlevel 1 (
  for %%V in (3.13 3.12 3.11 3.10) do call :try_launcher %%V
)

if not defined PYTHON_EXE (
  where python.exe >nul 2>nul
  if not errorlevel 1 call :try_python
)

if not defined PYTHON_EXE (
  echo.
  echo ERROR: Python 3.10, 3.11, 3.12, or 3.13 was not found.
  echo Install a 64-bit Python release from python.org and enable the
  echo "Add python.exe to PATH" option, then run this installer again.
  pause
  exit /b 1
)

echo Using:
"%PYTHON_EXE%" %PYTHON_ARGS% --version
if errorlevel 1 goto :failed

if exist "%OCULOSTUDIO_VENV%\" if not exist "%OCULOSTUDIO_PY%" (
  echo.
  echo Removing an incomplete runtime environment...
  rmdir /s /q "%OCULOSTUDIO_VENV%"
  if exist "%OCULOSTUDIO_VENV%\" goto :failed
)

if exist "%OCULOSTUDIO_PY%" (
  "%OCULOSTUDIO_PY%" "%~dp0tools\windows_python_check.py" >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Rebuilding an incompatible or damaged runtime environment...
    rmdir /s /q "%OCULOSTUDIO_VENV%"
    if exist "%OCULOSTUDIO_VENV%\" goto :failed
  )
)

if not exist "%OCULOSTUDIO_PY%" (
  echo.
  echo Creating the path-safe OculoStudio environment...
  if not exist "%OCULOSTUDIO_ENV_PARENT%\" mkdir "%OCULOSTUDIO_ENV_PARENT%"
  if errorlevel 1 goto :failed
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv "%OCULOSTUDIO_VENV%"
  if errorlevel 1 goto :failed
)

set "PY=%OCULOSTUDIO_PY%"
if not exist "%PY%" goto :failed

echo.
echo Checking pip...
"%PY%" -m ensurepip --upgrade >nul 2>nul
"%PY%" -m pip --version
if errorlevel 1 goto :failed

set "WHEEL=%~dp0dist\oculostudio-0.21.71-py3-none-any.whl"
if not exist "%WHEEL%" (
  echo.
  echo ERROR: The bundled OculoStudio wheel is missing:
  echo   %WHEEL%
  echo Re-extract the complete ZIP package.
  goto :failed
)

echo.
echo Installing OculoStudio runtime dependencies...
"%PY%" -m pip install --upgrade "%WHEEL%"
if errorlevel 1 goto :failed

rem Reinstall the local wheel itself even when this setup is run twice.
"%PY%" -m pip install --force-reinstall --no-deps "%WHEEL%"
if errorlevel 1 goto :failed

echo.
echo Installing the desktop shell...
set "OCULOSTUDIO_BROWSER_ONLY=0"
"%PY%" -m pip install --upgrade "pywebview>=5.1,<7"
if errorlevel 1 (
  set "OCULOSTUDIO_BROWSER_ONLY=1"
  >"%~dp0.browser_only" echo browser
  echo.
  echo Desktop WebView installation was not available.
  echo OculoStudio will use the same interface in the default browser.
) else (
  if exist "%~dp0.browser_only" del /q "%~dp0.browser_only" >nul 2>nul
)

echo.
echo Installing optional C3D and Parquet support when compatible wheels exist...
"%PY%" -m pip install "ezc3d>=1.5,<2" "pyarrow>=15,<24" >nul 2>nul

echo.
echo Checking the complete offline FFmpeg and ffprobe pair...
"%PY%" -m oculostudio media-tools --test-privacy --require-ffprobe --install-windows
if errorlevel 1 (
  echo.
  echo ERROR: Complete media-tool preflight failed. No video processing was started.
  echo Run REPAIR_MEDIA_TOOLS_WINDOWS.bat, or configure ffmpeg.exe in Studio.
  goto :failed
)
echo.
echo Both programs were tested: encoder AND frame timestamp reader.
echo Bundled tools are used without downloading another FFmpeg package.

echo.
echo Running the CleanCore self-check...
"%PY%" -m oculostudio check
if errorlevel 1 goto :failed

echo.
echo Setup completed successfully.
if "%OCULOSTUDIO_NO_AUTOSTART%"=="1" exit /b 0
if "%OCULOSTUDIO_BROWSER_ONLY%"=="1" (
  call "%~dp0START_WINDOWS_BROWSER.bat"
) else (
  call "%~dp0START_WINDOWS.bat"
)
exit /b %ERRORLEVEL%

:failed
echo.
echo ERROR: OculoStudio setup failed.
echo Review the first error above.
echo Runtime environment: %OCULOSTUDIO_VENV%
echo You can run REPAIR_INSTALLATION_WINDOWS.bat to rebuild only this generated environment.
pause
exit /b 1
