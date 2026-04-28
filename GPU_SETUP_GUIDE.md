# GPU Trainer Setup Guide

Train your BTC futures model on your local NVIDIA GPU and push predictions to your dashboard.

---

## What You Need

- Windows or Linux PC with NVIDIA GPU (RTX 3060 or better)
- Python 3.10 or 3.11
- NVIDIA drivers installed (version 525+)
- Your Replit dashboard URL

---

## Step 1: Copy the gpu_trainer folder to your PC

Download or copy the entire `gpu_trainer/` folder from this project to your local machine.

You can download it as a zip from the Replit file browser, or use git clone.

---

## Step 2: Run the setup script

Open a terminal/command prompt in the `gpu_trainer/` folder.

**Windows:**
```
setup_windows.bat
```

**Linux/Mac:**
```
chmod +x setup_linux.sh
./setup_linux.sh
```

This installs PyTorch with CUDA support and all dependencies. Takes about 2-5 minutes.

---

## Step 3: Train and push predictions

**Windows:**
```
venv\Scripts\activate.bat
python quick_start.py --url https://YOUR-APP.replit.app
```

**Linux/Mac:**
```
source venv/bin/activate
python quick_start.py --url https://YOUR-APP.replit.app
```

Replace `YOUR-APP` with your actual Replit app URL.

That's it! The script will:
1. Download your BTC data from the dashboard
2. Train the 5-head neural network on your GPU
3. Push a prediction to your dashboard

---

## Training Options

| Flag | Default | What it does |
|------|---------|-------------|
| `--epochs 50` | 50 | Number of training cycles |
| `--batch-size 64` | 64 | Samples per batch |
| `--lr 0.0001` | 0.0001 | Learning rate |
| `--predict-only` | off | Skip training, just predict from saved model |
| `--no-push` | off | Train but don't push prediction |

**Examples:**
```
# Train with more epochs for better accuracy
python quick_start.py --url https://YOUR-APP.replit.app --epochs 100

# Just make a new prediction from the last trained model
python quick_start.py --url https://YOUR-APP.replit.app --predict-only

# Train but don't send prediction yet
python quick_start.py --url https://YOUR-APP.replit.app --no-push
```

---

## Advanced Training (main.py)

For more control, use `main.py` directly:

```
# Fetch data from Binance directly (alternative to dashboard download)
python main.py fetch --candles 35000

# Train with specific flags
python main.py train --model enhanced_mlp --epochs 100 --multihead \
  --enable-quantile --enable-vol-state --enable-mu --enable-sigma \
  --focal-loss --pure-directional

# Start prediction API server
python main.py serve --port 8000
```

---

## What the Model Does

The EnhancedMultiHeadMLP has 5 output heads:

| Head | Purpose | Output |
|------|---------|--------|
| Classification | Direction signal | LONG / SHORT / HOLD with probabilities |
| Quantile | Return distribution | q10, q25, q50, q75, q90 price levels |
| VolState | Volatility regime | Contraction / Neutral / Expansion |
| Mu | Expected return | Predicted return over 16 bars (4 hours) |
| Sigma | Uncertainty | How confident the model is |

These combine into a complete trade plan with entry, stop loss, take profit, and position sizing.

---

## Troubleshooting

**"No GPU available"**
- Make sure NVIDIA drivers are installed: `nvidia-smi` should show your GPU
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

**"Cannot connect to dashboard"**
- Make sure your Replit app is running (visit the URL in a browser first)
- Check the URL is correct (include https://)

**"Not enough data"**
- Go to your dashboard's Neural Network tab
- Download at least 6 months of 15m BTC data

**Training is slow**
- Reduce batch size: `--batch-size 32`
- Reduce epochs: `--epochs 30`
- Check GPU is being used: should say "GPU: NVIDIA GeForce RTX 4070" at startup

**"CUDA out of memory"**
- Reduce batch size: `--batch-size 32` or `--batch-size 16`
