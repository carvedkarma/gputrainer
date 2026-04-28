#!/bin/bash
echo "============================================================"
echo "  BTC Futures GPU Trainer - Linux/Mac Setup"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Upgrading pip..."
pip install --upgrade pip

echo "[3/4] Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/4] Installing remaining dependencies..."
pip install -r requirements.txt

echo
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo
echo "Next step - run training:"
echo "  source venv/bin/activate"
echo "  python quick_start.py --url https://YOUR-APP.replit.app"
echo
