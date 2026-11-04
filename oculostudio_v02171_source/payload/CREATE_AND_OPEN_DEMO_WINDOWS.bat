@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0CREATE_PRIVACY_AOI_DEMO_WINDOWS.bat"
exit /b %ERRORLEVEL%
