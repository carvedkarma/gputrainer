# Local GPU Setup Guide

## Step 1: Install NVIDIA Drivers

### Windows
1. Download from https://www.nvidia.com/drivers
2. Install the latest Game Ready or Studio driver for your GPU

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
sudo reboot
```

Verify:
```bash
nvidia-smi
```

## Step 2: Install CUDA Toolkit

Download CUDA 12.1 from https://developer.nvidia.com/cuda-downloads

### Linux
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

Add to ~/.bashrc:
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### Windows
Run the installer and follow prompts.

Verify:
```bash
nvcc --version
```

## Step 3: Install cuDNN

1. Download from https://developer.nvidia.com/cudnn (requires NVIDIA account)
2. Extract and copy files to CUDA directory

### Linux
```bash
tar -xf cudnn-linux-x86_64-8.9.0.131_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## Step 4: Install Python Environment

```bash
# Create virtual environment
python -m venv btc_trader_venv
source btc_trader_venv/bin/activate  # Linux/Mac
# or: btc_trader_venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

Verify PyTorch GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## Step 5: Install Docker (Optional)

### Linux
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Step 6: Install PostgreSQL (if not using Docker)

### Linux
```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb btc_signals
sudo -u postgres createuser trader --pwprompt
```

### Windows
Download from https://www.postgresql.org/download/windows/

## Step 7: Configure Environment

Create `.env` file:
```bash
DATABASE_URL=postgresql://trader:your_password@localhost:5432/btc_signals
REDIS_URL=redis://localhost:6379
```

## Step 8: Run the System

### Without Docker
```bash
# Terminal 1: Start PostgreSQL (if not running as service)
sudo systemctl start postgresql

# Terminal 2: Fetch data and train
cd gpu_trainer
python main.py fetch --candles 50000
python main.py train --model transformer --epochs 100

# Terminal 3: Start prediction server
python main.py serve --port 8000

# Terminal 4: Monitor training
tensorboard --logdir logs
```

### With Docker
```bash
cd gpu_trainer
docker-compose up -d
docker-compose logs -f trainer
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in config.py
- Reduce model size (fewer layers/heads)
- Enable gradient checkpointing

### "No CUDA runtime" 
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### "cuDNN not found"
- Verify cuDNN installation
- Check LD_LIBRARY_PATH includes CUDA lib directory

### Docker GPU not working
- Install nvidia-container-toolkit
- Restart Docker service
- Check nvidia-smi works outside container first

## Performance Benchmarks (RTX 4070)

| Model | Parameters | Batch 64 | Batch 128 |
|-------|------------|----------|-----------|
| Transformer (6L) | 25M | 120 samples/sec | 180 samples/sec |
| LSTM (3L) | 8M | 200 samples/sec | 320 samples/sec |
| CNN (ResNet) | 12M | 250 samples/sec | 400 samples/sec |
| VAE | 15M | 180 samples/sec | 280 samples/sec |

Training time for 50,000 samples, 100 epochs: ~2-4 hours depending on model.
