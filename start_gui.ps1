# BTC Futures GPU Trainer - PowerShell Launcher
Write-Host ""
Write-Host "========================================"
Write-Host "  BTC Futures - GPU Neural Network Trainer"
Write-Host "========================================"
Write-Host ""

# Set the Replit proxy URL (change this to your deployed URL)
$env:REPLIT_PROXY_URL = "https://web-app-carvedkarma.replit.app"

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Starting GUI with proxy: $env:REPLIT_PROXY_URL"
Write-Host ""

# Launch the GUI
python gui.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start GUI"
    Write-Host ""
    Write-Host "Make sure you have:"
    Write-Host "  1. Python installed (python --version)"
    Write-Host "  2. Required packages (pip install -r requirements.txt)"
    Write-Host ""
    Read-Host "Press Enter to exit"
}
