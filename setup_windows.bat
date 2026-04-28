@echo off
echo ============================================================
echo   BTC Futures GPU Trainer - Windows Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo [3/4] Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo [4/4] Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next step - run training:
echo   venv\Scripts\activate.bat
echo   python quick_start.py --url https://YOUR-APP.replit.app
echo.
pause
