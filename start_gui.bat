@echo off
title BTC Futures GPU Trainer
echo.
echo ========================================
echo   BTC Futures - GPU Neural Network Trainer
echo ========================================
echo.
echo Starting GUI...
echo.

cd /d "%~dp0"
python gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start GUI
    echo.
    echo Make sure you have:
    echo   1. Python installed (python --version)
    echo   2. Required packages (pip install -r requirements.txt)
    echo.
    pause
)
