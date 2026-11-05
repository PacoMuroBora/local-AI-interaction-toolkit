@echo off
cd /d "%~dp0"
"C:\Users\Admin\.local\bin\uv.exe" run luzifer
if errorlevel 1 pause
