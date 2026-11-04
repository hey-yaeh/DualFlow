@echo off
rem Shared Windows runtime resolver for OculoStudio v0.21.64.
rem Do not use SETLOCAL here: callers need the variables defined below.

if not defined OCULOSTUDIO_APP_ROOT (
  for %%I in ("%~dp0..") do set "OCULOSTUDIO_APP_ROOT=%%~fI"
)

if not defined OCULOSTUDIO_ENV_PARENT (
  if defined LOCALAPPDATA (
    set "OCULOSTUDIO_ENV_PARENT=%LOCALAPPDATA%\OculoStudio\venvs"
  ) else if defined USERPROFILE (
    set "OCULOSTUDIO_ENV_PARENT=%USERPROFILE%\OculoStudio\venvs"
  ) else if defined TEMP (
    set "OCULOSTUDIO_ENV_PARENT=%TEMP%\OculoStudio\venvs"
  ) else (
    echo ERROR: Windows user storage could not be resolved.
    exit /b 2
  )
)

if not defined OCULOSTUDIO_VENV set "OCULOSTUDIO_VENV=%OCULOSTUDIO_ENV_PARENT%\v0_21_25"
set "OCULOSTUDIO_PY=%OCULOSTUDIO_VENV%\Scripts\python.exe"
set "OCULOSTUDIO_PYINSTALLER=%OCULOSTUDIO_VENV%\Scripts\pyinstaller.exe"
set "PYTHONUTF8=1"
exit /b 0
