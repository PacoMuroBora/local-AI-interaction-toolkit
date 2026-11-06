@echo off
cd /d "%~dp0"
echo Waiting 20s
timeout /t 20 /nobreak >nul
"C:\Users\Admin\.local\bin\uv.exe" run luzifer
if errorlevel 1 pause
