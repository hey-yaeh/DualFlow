@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0tools\windows_runtime.bat"
if not exist "%OCULOSTUDIO_PY%" (
  echo Run INSTALL_AND_RUN_WINDOWS.bat first.
  pause
  exit /b 1
)
echo Installing optional MotionStudio mesh backend...
"%OCULOSTUDIO_PY%" -m pip install --upgrade "torch>=2.2" "smplx>=0.1.28"
if errorlevel 1 goto :error
"%OCULOSTUDIO_PY%" -c "import torch, smplx; print('PASS: torch', torch.__version__, 'and smplx are available')"
if errorlevel 1 goto :error
echo.
echo MotionStudio bridge runtime is installed.
echo Configure Body Model Root in the MotionStudio panel, for example:
echo D:\DuoFlow\Body_Models
pause
exit /b 0
:error
echo.
echo Installation failed. A direct vertices_world cache can still be used without SMPL reconstruction.
pause
exit /b 1
