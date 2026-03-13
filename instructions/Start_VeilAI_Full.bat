@echo off
REM VeilAI Launcher - Full Featured Version
REM Double-click this file to start VeilAI with all features

cd /d "%~dp0..\..\python"

echo.
echo ========================================
echo   VeilAI - Image Protection System
echo   Full-Featured Version
echo ========================================
echo.
echo WARNING: This version requires all dependencies.
echo First run will download CLIP model (~350MB)
echo.
echo Starting full version...
echo It will open automatically in your browser.
echo.

python launch_ui.py --full

pause
