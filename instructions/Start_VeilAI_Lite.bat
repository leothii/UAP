@echo off
REM VeilAI Launcher - Lightweight Version (Recommended)
REM Double-click this file to start VeilAI

cd /d "%~dp0..\..\python"

echo.
echo ========================================
echo   VeilAI - Image Protection System
echo ========================================
echo.
echo Starting lightweight version...
echo It will open automatically in your browser.
echo.

python launch_ui.py --lite

pause
