@echo off
title Alpha Go Game Launcher
echo ============================================================
echo                  ALPHA GO GAME LAUNCHER
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if streamlit_app.py exists
if not exist "streamlit_app.py" (
    echo Error: streamlit_app.py not found!
    echo Please run this script from the game directory.
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error installing dependencies!
        pause
        exit /b 1
    )
)

echo.
echo Launching Alpha Go Game...
echo Game will open in your default web browser
echo Press Ctrl+C in this window to stop the game
echo.
echo ============================================================
echo.

REM Launch the Streamlit app
python -m streamlit run streamlit_app.py --server.headless false --browser.gatherUsageStats false

echo.
echo Game closed. Thanks for playing!
pause