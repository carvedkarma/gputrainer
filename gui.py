#!/usr/bin/env python3
"""
BTC Futures Trading - GPU Trainer Desktop GUI

A streamlined graphical interface for fetching data and training neural networks
on your local GPU. Optimized settings are applied automatically for best results.

Usage:
    python gui.py
    
    Or double-click gui.py on Windows
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import queue
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

# Optimal training defaults per model architecture (research-backed settings)
OPTIMAL_DEFAULTS = {
    "transformer": {"epochs": 150, "batch_size": 32, "lr": 0.0001, "desc": "Best for capturing complex temporal patterns"},
    "tft": {"epochs": 120, "batch_size": 32, "lr": 0.0001, "desc": "Temporal Fusion - interpretable attention"},
    "lstm": {"epochs": 100, "batch_size": 64, "lr": 0.001, "desc": "Classic RNN - fast and reliable"},
    "cnn": {"epochs": 80, "batch_size": 64, "lr": 0.001, "desc": "Fast pattern detection via convolutions"},
    "vae": {"epochs": 100, "batch_size": 32, "lr": 0.0005, "desc": "Variational autoencoder for regime detection"},
    "gnn": {"epochs": 120, "batch_size": 16, "lr": 0.0001, "desc": "Cross-asset correlation modeling"},
}

class LogRedirector:
    def __init__(self, widget, queue):
        self.widget = widget
        self.queue = queue
        
    def write(self, text):
        self.queue.put(text)
        
    def flush(self):
        pass

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 0:
        return "--:--"
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

class GPUTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BTC Futures - GPU Neural Network Trainer")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        
        self.configure_dark_theme()
        
        self.log_queue = queue.Queue()
        self.is_training = False
        self.is_fetching = False
        self.training_thread = None
        self.fetch_thread = None
        self.stop_training_flag = threading.Event()
        
        # Training state
        self.current_model = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss = None
        self.val_loss = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.models_completed = []
        
        # Live trade count tracking
        self.live_trade_count = {"total": 0, "long": 0, "short": 0, "session_start": None}
        
        # Label mode config (HOLD fix - 3 stages)
        # Options: "cost_aware" (Stage 1), "pure_directional" (Stage 2), "regime" (Stage 3)
        self.label_mode = "regime"  # Default to regime for best label distribution
        self.use_pure_directional = False  # Legacy: computed from label_mode
        self.use_regime_labels = True  # Legacy: computed from label_mode
        self.directional_threshold = 0.0020  # 0.20% for pure_directional mode
        self.trend_threshold = 0.0015  # 0.15% for trending regime
        self.range_threshold = 0.0030  # 0.30% for ranging regime
        self.min_confidence = 0.40  # Stage 1 min_confidence
        
        # Per-model training status for dashboard sync
        # Each model has its own oos_trades count that won't be overwritten by other models
        self.model_status = {
            "transformer": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
            "tft": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
            "lstm": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
            "cnn": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
            "vae": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
            "gnn": {"status": "pending", "accuracy": None, "loss": None, "epochs": 0, "best_epoch": 0, "oos_trades": 0, "oos_epoch": 0},
        }
        
        # Current model being trained (for OOS trade tracking)
        self.current_training_model = None
        
        # Mapping from checkpoint filename patterns to standardized model types
        self.model_type_patterns = {
            # Multi-head models (check first - have _multihead suffix) and legacy models
            "transformer": ["best_transformer_multihead", "transformer_multihead", "multihead_transformer", "transformer_price", "transformer", "best_transformer"],
            "tft": ["best_tft_multihead", "tft_multihead", "multihead_tft", "temporal_fusion_transformer", "tft", "best_temporal_fusion", "best_tft"],
            "lstm": ["best_lstm_multihead", "lstm_multihead", "multihead_lstm", "bidirectional_lstm", "lstm", "stacked_lstm", "conv_lstm", "best_lstm", "best_bidirectional"],
            "cnn": ["best_cnn_multihead", "cnn_multihead", "multihead_cnn", "resnet_price", "resnet", "cnn", "inception", "wavenet", "best_resnet", "best_cnn"],
            "vae": ["best_vae_multihead", "vae_multihead", "multihead_vae", "market_vae", "vae", "conditional_vae", "best_vae", "best_market_vae"],
            "gnn": ["best_gnn_multihead", "gnn_multihead", "multihead_gnn", "cross_asset_gnn", "temporal_gnn", "gnn", "best_gnn", "best_cross_asset"],
        }
        
        # Scan for existing checkpoints on startup
        self.scan_existing_checkpoints()
        
        # GPU state
        self.gpu_name = None
        self.gpu_memory_used = 0
        self.gpu_memory_total = 0
        self.gpu_utilization = 0
        
        # Timing state
        self.fetch_start_time = None
        self.fetch_total_items = 0
        self.fetch_completed_items = 0
        self.training_start_time = None
        self.epoch_times = []
        
        # Saved data state
        self.saved_data_info = {}
        self.data_loaded = False
        
        self.create_widgets()
        self.wire_stdout_to_log()
        self.check_gpu_status()
        self.process_log_queue()
        self.start_status_push()
        self.start_gpu_monitor()
        
        # Check for saved data on startup
        self.root.after(500, self.check_saved_data)
    
    def wire_stdout_to_log(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = LogRedirector(self.log_text, self.log_queue)
        sys.stderr = LogRedirector(self.log_text, self.log_queue)
        
    def configure_dark_theme(self):
        self.colors = {
            'bg': '#0f0f1a',
            'bg_secondary': '#1a1a2e',
            'bg_tertiary': '#16213e',
            'bg_card': '#1e1e32',
            'accent': '#00d4aa',
            'accent_hover': '#00f5c4',
            'accent_dim': '#007a63',
            'text': '#f0f0f0',
            'text_secondary': '#a0a0b0',
            'text_tertiary': '#606070',
            'success': '#00d26a',
            'warning': '#ffd93d',
            'error': '#ff4757',
            'border': '#2a2a4a',
            'progress_bg': '#252540',
            'progress_fill': '#00d4aa'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['bg_card'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Segoe UI', 10))
        style.configure('Card.TLabel', background=self.colors['bg_card'], foreground=self.colors['text'])
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), foreground=self.colors['accent'])
        style.configure('Title.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.colors['text'])
        style.configure('Status.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_secondary'])
        style.configure('Value.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['accent'])
        style.configure('Dim.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_tertiary'])
        
        style.configure('TButton', 
                       background=self.colors['accent'],
                       foreground='#000000',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(20, 10))
        style.map('TButton',
                 background=[('active', self.colors['accent_hover']), ('disabled', self.colors['border'])])
        
        style.configure('Secondary.TButton',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 9),
                       padding=(10, 6))
        style.map('Secondary.TButton',
                 background=[('active', self.colors['border']), ('disabled', self.colors['bg_secondary'])])
        
        style.configure('Stop.TButton',
                       background=self.colors['error'],
                       foreground='#ffffff',
                       font=('Segoe UI', 9, 'bold'),
                       padding=(10, 6))
        style.map('Stop.TButton', background=[('active', '#ff6b7a')])
        
        style.configure('TCombobox',
                       fieldbackground=self.colors['bg_tertiary'],
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['text'],
                       arrowcolor=self.colors['text'])
        
        style.configure('Horizontal.TProgressbar',
                       background=self.colors['progress_fill'],
                       troughcolor=self.colors['progress_bg'],
                       thickness=8)
        
        style.configure('TLabelframe', background=self.colors['bg_card'])
        style.configure('TLabelframe.Label', background=self.colors['bg_card'], foreground=self.colors['text'], font=('Segoe UI', 10, 'bold'))
        
    def create_widgets(self):
        main_container = ttk.Frame(self.root, padding=15)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header with GPU status
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="GPU Neural Network Trainer", style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # GPU status badge
        self.gpu_badge_frame = tk.Frame(header_frame, bg=self.colors['bg_tertiary'], padx=10, pady=5)
        self.gpu_badge_frame.pack(side=tk.RIGHT)
        self.gpu_status_label = tk.Label(self.gpu_badge_frame, text="Detecting GPU...", 
                                          bg=self.colors['bg_tertiary'], fg=self.colors['text_secondary'],
                                          font=('Segoe UI', 9))
        self.gpu_status_label.pack()
        
        # TRAINING HEALTH ALERT BANNER - Shows when training issues detected
        # Hidden by default, shown when health monitor detects problems
        self.alert_banner_frame = tk.Frame(main_container, bg='#ff4757', padx=12, pady=8)
        # Don't pack yet - will be shown when alerts occur
        
        self.alert_icon_label = tk.Label(self.alert_banner_frame, text="⚠", 
                                          bg='#ff4757', fg='#ffffff',
                                          font=('Segoe UI', 14, 'bold'))
        self.alert_icon_label.pack(side=tk.LEFT, padx=(0, 8))
        
        self.alert_message_label = tk.Label(self.alert_banner_frame, text="Training issue detected!",
                                             bg='#ff4757', fg='#ffffff',
                                             font=('Segoe UI', 10, 'bold'), anchor=tk.W)
        self.alert_message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.alert_dismiss_btn = tk.Button(self.alert_banner_frame, text="✕", 
                                            bg='#ff4757', fg='#ffffff',
                                            font=('Segoe UI', 10, 'bold'),
                                            relief=tk.FLAT, cursor='hand2',
                                            command=self.dismiss_alert)
        self.alert_dismiss_btn.pack(side=tk.RIGHT)
        
        # Track alert state
        self.current_alerts = []
        self.alert_visible = False
        
        # Content area - two columns
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Controls and Stats
        left_panel = ttk.Frame(content_frame, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        self.create_data_panel(left_panel)
        self.create_training_panel(left_panel)
        self.create_gpu_stats_panel(left_panel)
        
        # Right column - Log output
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_log_panel(right_panel)
        
    def create_data_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Data Download ", padding=12)
        frame.pack(fill=tk.X, pady=(0, 12))
        
        # Proxy URL
        proxy_frame = ttk.Frame(frame)
        proxy_frame.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(proxy_frame, text="Replit Server URL:", style='Card.TLabel').pack(anchor=tk.W)
        self.proxy_url_var = tk.StringVar(value=os.getenv("REPLIT_PROXY_URL", "https://99f68291-4a03-450a-9815-ebee9435cee2-00-2os5ge21n6uho.spock.replit.dev"))
        proxy_entry = tk.Entry(proxy_frame, textvariable=self.proxy_url_var, 
                               bg=self.colors['bg_tertiary'], fg=self.colors['text'],
                               insertbackground=self.colors['text'], font=('Segoe UI', 9),
                               relief=tk.FLAT, highlightthickness=1, highlightbackground=self.colors['border'])
        proxy_entry.pack(fill=tk.X, pady=(4, 0), ipady=6)
        
        # Data info label
        info_label = ttk.Label(frame, text="Downloads 15m training data from Replit\n(BTCUSDT only - matches training config)", 
                               style='Dim.TLabel')
        info_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Saved data status panel
        self.saved_data_frame = tk.Frame(frame, bg=self.colors['bg_tertiary'], padx=10, pady=8)
        self.saved_data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.saved_data_label = tk.Label(self.saved_data_frame, text="Checking for saved data...",
                                          bg=self.colors['bg_tertiary'], fg=self.colors['text_secondary'],
                                          font=('Segoe UI', 9), anchor=tk.W, justify=tk.LEFT)
        self.saved_data_label.pack(fill=tk.X)
        
        self.saved_data_details = tk.Label(self.saved_data_frame, text="",
                                            bg=self.colors['bg_tertiary'], fg=self.colors['text_tertiary'],
                                            font=('Segoe UI', 8), anchor=tk.W, justify=tk.LEFT)
        self.saved_data_details.pack(fill=tk.X, pady=(2, 0))
        
        # Progress section
        self.fetch_progress_frame = ttk.Frame(frame)
        self.fetch_progress_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.fetch_progress = ttk.Progressbar(self.fetch_progress_frame, mode='determinate', length=300)
        self.fetch_progress.pack(fill=tk.X)
        
        # Stats row (hidden until fetching)
        self.fetch_stats_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        self.fetch_stats_frame.pack(fill=tk.X, pady=(4, 8))
        
        self.fetch_pct_label = tk.Label(self.fetch_stats_frame, text="0%", 
                                         bg=self.colors['bg_card'], fg=self.colors['accent'],
                                         font=('Segoe UI', 10, 'bold'))
        self.fetch_pct_label.pack(side=tk.LEFT)
        
        self.fetch_eta_label = tk.Label(self.fetch_stats_frame, text="", 
                                         bg=self.colors['bg_card'], fg=self.colors['text_secondary'],
                                         font=('Segoe UI', 9))
        self.fetch_eta_label.pack(side=tk.RIGHT)
        
        self.fetch_speed_label = tk.Label(self.fetch_stats_frame, text="", 
                                           bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                           font=('Segoe UI', 9))
        self.fetch_speed_label.pack(side=tk.RIGHT, padx=(0, 15))
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        self.load_saved_btn = ttk.Button(btn_frame, text="Load Saved", command=self.load_saved_data, 
                                          style='Secondary.TButton', state=tk.DISABLED)
        self.load_saved_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.fetch_btn = ttk.Button(btn_frame, text="Download All Data", command=self.start_fetch)
        self.fetch_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def create_training_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Neural Network Training ", padding=12)
        frame.pack(fill=tk.X, pady=(0, 12))
        
        # Model selection
        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(model_frame, text="Architecture:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar(value="transformer")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                    values=list(OPTIMAL_DEFAULTS.keys()), width=15, state='readonly')
        model_combo.pack(side=tk.LEFT, padx=(10, 0))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        # Model description
        self.model_desc_label = tk.Label(frame, text=OPTIMAL_DEFAULTS["transformer"]["desc"],
                                          bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                          font=('Segoe UI', 9), wraplength=340, justify=tk.LEFT)
        self.model_desc_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Timeframe mode - FIXED to 15m only (matches Replit data and training config)
        tf_frame = ttk.Frame(frame)
        tf_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(tf_frame, text="Timeframe:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.timeframe_var = tk.StringVar(value="15m")  # Fixed to 15m only
        tf_label = ttk.Label(tf_frame, text="15m (fixed)", style='Card.TLabel')
        tf_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Timeframe description
        self.tf_desc_label = tk.Label(frame, text="15m only: matches Replit data & training config",
                                       bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                       font=('Segoe UI', 9), wraplength=340, justify=tk.LEFT)
        self.tf_desc_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Asset selection - FIXED to BTCUSDT only (matches Replit data)
        asset_frame = ttk.Frame(frame)
        asset_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(asset_frame, text="Asset:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.asset_vars = {}
        # Only BTCUSDT is supported - show as fixed label
        btc_label = ttk.Label(asset_frame, text="BTCUSDT (fixed)", style='Card.TLabel')
        btc_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Keep asset_vars for compatibility but only BTC
        for asset in ["BTCUSDT"]:
            var = tk.BooleanVar(value=True)
            self.asset_vars[asset] = var
        
        # Cost mode selection
        cost_frame = ttk.Frame(frame)
        cost_frame.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(cost_frame, text="Cost Mode:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.cost_mode_var = tk.StringVar(value="taker_taker")
        cost_combo = ttk.Combobox(cost_frame, textvariable=self.cost_mode_var,
                                   values=["taker_taker", "maker_taker", "maker_maker"], 
                                   width=12, state='readonly')
        cost_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Cost display label
        self.cost_display = tk.Label(cost_frame, text="(0.09% round-trip)",
                                      bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                      font=('Segoe UI', 9))
        self.cost_display.pack(side=tk.LEFT, padx=(8, 0))
        cost_combo.bind('<<ComboboxSelected>>', self.on_cost_mode_changed)
        
        # Multi-head mode checkbox
        multihead_frame = ttk.Frame(frame)
        multihead_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.multihead_var = tk.BooleanVar(value=True)  # Default to multi-head mode
        multihead_check = ttk.Checkbutton(
            multihead_frame, 
            text="Multi-Head Training (Classification + Regression + Quantile)",
            variable=self.multihead_var
        )
        multihead_check.pack(side=tk.LEFT)
        
        multihead_info = tk.Label(multihead_frame, text="(recommended)",
                                   bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                   font=('Segoe UI', 9))
        multihead_info.pack(side=tk.LEFT, padx=(8, 0))
        
        # Label Mode selection (HOLD fix stages)
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(label_frame, text="Label Mode:", style='Card.TLabel').pack(side=tk.LEFT)
        
        self.label_mode_var = tk.StringVar(value="regime")
        label_combo = ttk.Combobox(label_frame, textvariable=self.label_mode_var,
                                    values=["cost_aware", "pure_directional", "regime"], 
                                    width=15, state='readonly')
        label_combo.pack(side=tk.LEFT, padx=(10, 0))
        label_combo.bind('<<ComboboxSelected>>', self.on_label_mode_changed)
        
        # Label mode description
        self.label_desc_label = tk.Label(frame, text="Regime: ADX-based adaptive thresholds (best for label balance)",
                                          bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                          font=('Segoe UI', 9), wraplength=340, justify=tk.LEFT)
        self.label_desc_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Auto-settings display
        settings_frame = tk.Frame(frame, bg=self.colors['bg_tertiary'], padx=10, pady=8)
        settings_frame.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(settings_frame, text="Optimized Settings:", 
                  font=('Segoe UI', 9), foreground=self.colors['text_secondary'],
                  background=self.colors['bg_tertiary']).pack(anchor=tk.W)
        
        self.settings_display = tk.Label(settings_frame, text="Epochs: 150 | Batch: 32 | LR: 0.0001",
                                          bg=self.colors['bg_tertiary'], fg=self.colors['accent'],
                                          font=('Segoe UI', 10, 'bold'))
        self.settings_display.pack(anchor=tk.W, pady=(2, 0))
        
        # Progress section
        self.train_progress = ttk.Progressbar(frame, mode='determinate', length=300)
        self.train_progress.pack(fill=tk.X, pady=(0, 4))
        
        # Training stats row
        self.train_stats_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        self.train_stats_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.train_pct_label = tk.Label(self.train_stats_frame, text="Ready", 
                                         bg=self.colors['bg_card'], fg=self.colors['text_secondary'],
                                         font=('Segoe UI', 10, 'bold'))
        self.train_pct_label.pack(side=tk.LEFT)
        
        self.train_eta_label = tk.Label(self.train_stats_frame, text="", 
                                         bg=self.colors['bg_card'], fg=self.colors['text_secondary'],
                                         font=('Segoe UI', 9))
        self.train_eta_label.pack(side=tk.RIGHT)
        
        # Loss display
        self.loss_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        self.loss_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_loss_label = tk.Label(self.loss_frame, text="Train: --", 
                                          bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                          font=('Segoe UI', 9))
        self.train_loss_label.pack(side=tk.LEFT)
        
        self.val_loss_label = tk.Label(self.loss_frame, text="Val: --", 
                                        bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                        font=('Segoe UI', 9))
        self.val_loss_label.pack(side=tk.LEFT, padx=(15, 0))
        
        self.best_label = tk.Label(self.loss_frame, text="", 
                                    bg=self.colors['bg_card'], fg=self.colors['success'],
                                    font=('Segoe UI', 9))
        self.best_label.pack(side=tk.RIGHT)
        
        # OOS Trade Count Display (shown during training monitoring sweeps)
        self.trade_count_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        self.trade_count_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.trade_count_title = tk.Label(self.trade_count_frame, text="OOS Trades: ", 
                                          bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                          font=('Segoe UI', 9))
        self.trade_count_title.pack(side=tk.LEFT)
        
        self.trade_total_label = tk.Label(self.trade_count_frame, text="--", 
                                          bg=self.colors['bg_card'], fg=self.colors['accent'],
                                          font=('Segoe UI', 9, 'bold'))
        self.trade_total_label.pack(side=tk.LEFT)
        
        # MIN_TRADES threshold indicator (30 required for policy eligibility)
        self.trade_threshold_label = tk.Label(self.trade_count_frame, text=" / 30 min", 
                                              bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                              font=('Segoe UI', 8))
        self.trade_threshold_label.pack(side=tk.LEFT)
        
        self.trade_status_label = tk.Label(self.trade_count_frame, text="", 
                                           bg=self.colors['bg_card'], fg=self.colors['warning'],
                                           font=('Segoe UI', 8, 'bold'))
        self.trade_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        self.trade_epoch_label = tk.Label(self.trade_count_frame, text="", 
                                          bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                          font=('Segoe UI', 8))
        self.trade_epoch_label.pack(side=tk.RIGHT)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        self.train_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.stop_train_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_training, 
                                          style='Stop.TButton', state=tk.DISABLED)
        self.stop_train_btn.pack(side=tk.LEFT)
        
        # Quick action
        ttk.Button(frame, text="Train All 6 Models", command=self.train_all_models,
                   style='Secondary.TButton').pack(fill=tk.X, pady=(10, 0))
        
    def create_gpu_stats_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" GPU Status ", padding=12)
        frame.pack(fill=tk.X, pady=(0, 12))
        
        # GPU name
        self.gpu_name_label = tk.Label(frame, text="Detecting...",
                                        bg=self.colors['bg_card'], fg=self.colors['text'],
                                        font=('Segoe UI', 10, 'bold'))
        self.gpu_name_label.pack(anchor=tk.W)
        
        # VRAM usage
        vram_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        vram_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(vram_frame, text="VRAM", bg=self.colors['bg_card'], 
                 fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        self.vram_progress = ttk.Progressbar(vram_frame, mode='determinate', length=300)
        self.vram_progress.pack(fill=tk.X, pady=(2, 0))
        
        self.vram_label = tk.Label(vram_frame, text="-- / -- GB",
                                    bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                    font=('Segoe UI', 9))
        self.vram_label.pack(anchor=tk.E)
        
        # Utilization
        util_frame = tk.Frame(frame, bg=self.colors['bg_card'])
        util_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Label(util_frame, text="Utilization", bg=self.colors['bg_card'], 
                 fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        self.util_progress = ttk.Progressbar(util_frame, mode='determinate', length=300)
        self.util_progress.pack(fill=tk.X, pady=(2, 0))
        
        self.util_label = tk.Label(util_frame, text="--%",
                                    bg=self.colors['bg_card'], fg=self.colors['text_tertiary'],
                                    font=('Segoe UI', 9))
        self.util_label.pack(anchor=tk.E)
        
    def create_log_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=" Output Log ", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            font=('Consolas', 9),
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Clear", command=self.clear_log, style='Secondary.TButton').pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save Log", command=self.save_log, style='Secondary.TButton').pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(btn_frame, text="Start API Server", command=self.start_api_server, style='Secondary.TButton').pack(side=tk.RIGHT)
        
        self.log("=" * 55)
        self.log("  BTC Futures GPU Trainer - Ready")
        self.log("  Optimal settings applied automatically per model")
        self.log("=" * 55)
        self.log("")
        
    def on_model_changed(self, event=None):
        model = self.model_var.get()
        defaults = OPTIMAL_DEFAULTS.get(model, OPTIMAL_DEFAULTS["transformer"])
        
        self.model_desc_label.config(text=defaults["desc"])
        self.settings_display.config(text=f"Epochs: {defaults['epochs']} | Batch: {defaults['batch_size']} | LR: {defaults['lr']}")
    
    def on_cost_mode_changed(self, event=None):
        mode = self.cost_mode_var.get()
        cost_map = {
            "taker_taker": ("0.09%", 0.0009),
            "maker_taker": ("0.06%", 0.0006),
            "maker_maker": ("0.04%", 0.0004)
        }
        display, _ = cost_map.get(mode, ("0.09%", 0.0009))
        self.cost_display.config(text=f"({display} round-trip)")
    
    def on_label_mode_changed(self, event=None):
        """Handle label mode dropdown change."""
        mode = self.label_mode_var.get()
        self.label_mode = mode
        
        # Update computed legacy flags
        self.use_pure_directional = (mode == "pure_directional")
        self.use_regime_labels = (mode == "regime")
        
        # Update description
        desc_map = {
            "cost_aware": "Stage 1: min_confidence=0.40 gating (may still be HOLD-heavy)",
            "pure_directional": "Stage 2: Simple return threshold (0.20% default)",
            "regime": "Stage 3: ADX-based adaptive thresholds (best for label balance)"
        }
        self.label_desc_label.config(text=desc_map.get(mode, ""))
    
    def on_timeframe_changed(self, event=None):
        # Timeframe is now fixed to 15m only
        self.tf_desc_label.config(text="15m only: matches Replit data & training config")
    
    def select_all_assets(self):
        for var in self.asset_vars.values():
            var.set(True)
    
    def get_selected_assets(self):
        """Get list of selected asset symbols"""
        return [asset for asset, var in self.asset_vars.items() if var.get()]
    
    def get_timeframe_mode(self):
        """Get timeframe mode: always '15m' (fixed to match Replit data)"""
        return "15m"  # Fixed to 15m only
        
    def get_trading_cost(self) -> float:
        """Get the selected trading cost for label creation"""
        mode = self.cost_mode_var.get()
        cost_map = {
            "taker_taker": 0.0009,
            "maker_taker": 0.0006,
            "maker_maker": 0.0004
        }
        return cost_map.get(mode, 0.0009)
        
    def check_gpu_status(self):
        def check():
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.gpu_name = name
                    self.gpu_memory_total = mem_total
                    
                    short_name = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                    status = f"✓ {short_name}"
                    color = self.colors['success']
                    
                    self.log(f"GPU Detected: {name}")
                    self.log(f"VRAM: {mem_total:.1f} GB")
                    self.log("")
                    
                    self.root.after(0, lambda: self.gpu_name_label.config(text=f"{short_name} ({mem_total:.1f} GB)"))
                else:
                    status = "⚠ No GPU - CPU Mode"
                    color = self.colors['warning']
                    self.log("WARNING: No GPU detected. Training will be slow.")
                    self.root.after(0, lambda: self.gpu_name_label.config(text="CPU Mode (No GPU)"))
                    
            except ImportError:
                status = "✗ PyTorch Missing"
                color = self.colors['error']
                self.log("ERROR: PyTorch not installed")
            except Exception as e:
                status = f"✗ Error"
                color = self.colors['error']
                self.log(f"GPU check error: {e}")
                
            self.root.after(0, lambda: [
                self.gpu_status_label.config(text=status, fg=color),
                self.gpu_badge_frame.config(bg=self.colors['bg_tertiary'])
            ])
            
        threading.Thread(target=check, daemon=True).start()
        
    def start_gpu_monitor(self):
        def monitor():
            while True:
                try:
                    import torch
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated(0) / 1024**3
                        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        mem_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        
                        self.gpu_memory_used = mem_used
                        self.gpu_memory_total = mem_total
                        
                        # Try to get utilization (requires pynvml)
                        util_pct = 0
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            util_pct = util.gpu
                            self.gpu_utilization = util_pct
                        except:
                            pass
                        
                        self.root.after(0, lambda m=mem_used, t=mem_total, p=mem_pct, u=util_pct: self._update_gpu_display(m, t, p, u))
                except:
                    pass
                time.sleep(2)
                
        threading.Thread(target=monitor, daemon=True).start()
        
    def _update_gpu_display(self, mem_used, mem_total, mem_pct, util_pct):
        self.vram_progress['value'] = mem_pct
        self.vram_label.config(text=f"{mem_used:.1f} / {mem_total:.1f} GB")
        
        self.util_progress['value'] = util_pct
        self.util_label.config(text=f"{util_pct}%")
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}\n")
        
    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)
        
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def show_training_alert(self, severity: str, message: str, details: dict = None):
        """
        Show a training health alert banner.
        
        Called by the training health monitor when issues are detected.
        
        Args:
            severity: 'CRITICAL', 'WARNING', or 'INFO'
            message: Alert message to display
            details: Optional dict with additional details
        """
        # Set colors based on severity
        if severity == 'CRITICAL':
            bg_color = '#ff4757'  # Red
            icon = "⛔"
        elif severity == 'WARNING':
            bg_color = '#ffa502'  # Orange
            icon = "⚠"
        else:
            bg_color = '#3498db'  # Blue
            icon = "ℹ"
        
        # Update banner appearance
        self.alert_banner_frame.config(bg=bg_color)
        self.alert_icon_label.config(bg=bg_color, text=icon)
        self.alert_message_label.config(bg=bg_color, text=message)
        self.alert_dismiss_btn.config(bg=bg_color)
        
        # Track the alert
        alert_info = {
            'severity': severity,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.current_alerts.append(alert_info)
        
        # Show the banner if not already visible
        if not self.alert_visible:
            self.alert_banner_frame.pack(fill=tk.X, pady=(0, 10), before=self.alert_banner_frame.master.winfo_children()[2])
            self.alert_visible = True
        
        # Log the alert
        self.log(f"[HEALTH {severity}] {message}")
        
        # Play alert sound for critical issues
        if severity == 'CRITICAL':
            try:
                self.root.bell()
            except:
                pass
    
    def dismiss_alert(self):
        """Dismiss the current alert banner."""
        if self.alert_visible:
            self.alert_banner_frame.pack_forget()
            self.alert_visible = False
            self.current_alerts.clear()
    
    def handle_health_alert_callback(self, severity: str, message: str, details: dict):
        """
        Callback for the TrainingHealthMonitor to send alerts to GUI.
        
        This is passed to the trainer and called when training issues are detected.
        Thread-safe: schedules update on main thread.
        """
        # Schedule on main thread since this may be called from training thread
        self.root.after(0, lambda: self.show_training_alert(severity, message, details))
        
    def save_log(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log(f"Log saved to: {filename}")
    
    def scan_existing_checkpoints(self):
        """Scan checkpoints directory for trained models and update model_status."""
        try:
            checkpoint_dir = Path(__file__).parent / "checkpoints"
            if not checkpoint_dir.exists():
                return
            
            # Find best_* checkpoint files
            best_checkpoints = list(checkpoint_dir.glob("best_*.pt"))
            if not best_checkpoints:
                return
            
            for ckpt_path in best_checkpoints:
                try:
                    filename = ckpt_path.stem.lower()
                    
                    # Map filename to model type
                    model_type = None
                    for mtype, patterns in self.model_type_patterns.items():
                        for pattern in patterns:
                            if pattern in filename:
                                model_type = mtype
                                break
                        if model_type:
                            break
                    
                    if model_type and model_type in self.model_status:
                        # Try to load checkpoint metadata
                        import torch
                        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                        
                        accuracy = checkpoint.get('val_accuracy', 0)
                        epoch = checkpoint.get('epoch', 0)
                        loss = checkpoint.get('val_loss', None)
                        
                        # Preserve oos_trades when updating status from checkpoint scan
                        existing_oos_trades = self.model_status.get(model_type, {}).get("oos_trades", 0)
                        existing_oos_epoch = self.model_status.get(model_type, {}).get("oos_epoch", 0)
                        self.model_status[model_type] = {
                            "status": "complete",
                            "accuracy": accuracy,
                            "loss": loss,
                            "epochs": epoch,
                            "best_epoch": epoch,
                            "oos_trades": existing_oos_trades,
                            "oos_epoch": existing_oos_epoch
                        }
                        
                        if model_type not in self.models_completed:
                            self.models_completed.append(model_type)
                        
                        print(f"[Checkpoint] Found trained model: {ckpt_path.name} -> {model_type} (acc={accuracy:.1f}%)")
                        
                except Exception as e:
                    print(f"[Checkpoint] Error loading {ckpt_path.name}: {e}")
                    
        except Exception as e:
            print(f"[Checkpoint] Error scanning checkpoints: {e}")
    
    def check_saved_data(self):
        """Check for existing parquet files on startup"""
        def scan():
            try:
                data_dir = Path(__file__).parent / "data_cache"
                if not data_dir.exists():
                    self.root.after(0, lambda: self._update_saved_data_ui(None))
                    return
                
                parquet_files = list(data_dir.glob("*.parquet"))
                if not parquet_files:
                    self.root.after(0, lambda: self._update_saved_data_ui(None))
                    return
                
                # Scan each file for stats
                import pandas as pd
                total_candles = 0
                symbols = set()
                timeframes = set()
                oldest_ts = None
                newest_ts = None
                newest_modified = None
                file_details = []
                
                for pf in parquet_files:
                    try:
                        # Parse filename: BTCUSDT_15m.parquet
                        name = pf.stem
                        parts = name.split('_')
                        if len(parts) >= 2:
                            sym = parts[0]
                            tf = parts[1]
                            symbols.add(sym)
                            timeframes.add(tf)
                        
                        # Get file modification time
                        mtime = pf.stat().st_mtime
                        if newest_modified is None or mtime > newest_modified:
                            newest_modified = mtime
                        
                        # Read parquet metadata only (fast)
                        df = pd.read_parquet(pf)
                        count = len(df)
                        total_candles += count
                        
                        if 'timestamp' in df.columns and count > 0:
                            min_ts = df['timestamp'].min()
                            max_ts = df['timestamp'].max()
                            if oldest_ts is None or min_ts < oldest_ts:
                                oldest_ts = min_ts
                            if newest_ts is None or max_ts > newest_ts:
                                newest_ts = max_ts
                        
                        file_details.append(f"{name}: {count:,}")
                    except Exception as e:
                        pass
                
                info = {
                    'total_candles': total_candles,
                    'symbols': sorted(symbols),
                    'timeframes': sorted(timeframes),
                    'oldest_ts': oldest_ts,
                    'newest_ts': newest_ts,
                    'newest_modified': newest_modified,
                    'file_count': len(parquet_files),
                    'file_details': file_details
                }
                
                self.saved_data_info = info
                self.root.after(0, lambda: self._update_saved_data_ui(info))
                
            except Exception as e:
                self.root.after(0, lambda: self._update_saved_data_ui(None))
        
        threading.Thread(target=scan, daemon=True).start()
    
    def _update_saved_data_ui(self, info):
        """Update the saved data panel in UI"""
        if info is None or info.get('total_candles', 0) == 0:
            self.saved_data_label.config(
                text="No saved data found",
                fg=self.colors['text_tertiary']
            )
            self.saved_data_details.config(text="")
            self.load_saved_btn.config(state=tk.DISABLED)
        else:
            # Format the info
            candles = info['total_candles']
            files = info['file_count']
            symbols = ', '.join(info['symbols'][:4])
            timeframes = ', '.join(info['timeframes'][:5])
            
            # Calculate data age
            modified = info.get('newest_modified')
            if modified:
                age = datetime.now() - datetime.fromtimestamp(modified)
                if age.days > 0:
                    age_str = f"{age.days}d ago"
                elif age.seconds > 3600:
                    age_str = f"{age.seconds // 3600}h ago"
                else:
                    age_str = f"{age.seconds // 60}m ago"
            else:
                age_str = "Unknown"
            
            # Format date range
            if info.get('oldest_ts') and info.get('newest_ts'):
                try:
                    start = datetime.fromtimestamp(info['oldest_ts'] / 1000).strftime('%Y-%m-%d')
                    end = datetime.fromtimestamp(info['newest_ts'] / 1000).strftime('%Y-%m-%d')
                    date_range = f"{start} to {end}"
                except:
                    date_range = "N/A"
            else:
                date_range = "N/A"
            
            self.saved_data_label.config(
                text=f"Saved: {candles:,} candles ({files} files) - Updated {age_str}",
                fg=self.colors['success']
            )
            self.saved_data_details.config(
                text=f"{symbols} | {timeframes} | {date_range}"
            )
            self.load_saved_btn.config(state=tk.NORMAL)
    
    def load_saved_data(self):
        """Load saved parquet files for training without re-downloading"""
        if self.data_loaded:
            self.log("Data already loaded!")
            return
        
        def do_load():
            try:
                data_dir = Path(__file__).parent / "data_cache"
                parquet_files = list(data_dir.glob("*.parquet"))
                
                self.log("")
                self.log("=" * 55)
                self.log("  LOADING SAVED DATA FROM DISK")
                self.log("=" * 55)
                self.log("")
                
                import pandas as pd
                total_candles = 0
                
                for pf in parquet_files:
                    try:
                        df = pd.read_parquet(pf)
                        total_candles += len(df)
                        self.log(f"  Loaded {len(df):,} candles: {pf.name}")
                    except Exception as e:
                        self.log(f"  Error loading {pf.name}: {e}")
                
                self.log("")
                self.log(f"Total: {total_candles:,} candles loaded from disk")
                self.log("Ready for training!")
                self.log("")
                
                self.data_loaded = True
                self.root.after(0, lambda: self._on_data_loaded())
                
            except Exception as e:
                self.log(f"Error loading saved data: {e}")
        
        threading.Thread(target=do_load, daemon=True).start()
    
    def _on_data_loaded(self):
        """Update UI after data is loaded"""
        self.load_saved_btn.config(text="Data Loaded", state=tk.DISABLED)
        self.saved_data_label.config(
            text=f"Data loaded and ready for training",
            fg=self.colors['accent']
        )
    
    def start_status_push(self):
        def push_loop():
            while True:
                self.push_status_to_replit()
                time.sleep(5)
        
        threading.Thread(target=push_loop, daemon=True).start()
        
    def push_status_to_replit(self):
        try:
            import requests
            proxy_url = self.proxy_url_var.get().strip()
            if not proxy_url:
                return
            
            # Try to get training mode from local API server
            training_mode = None
            training_mode_description = None
            input_dim = None
            
            try:
                # Query local FastAPI server for models status (includes training mode)
                local_status = requests.get("http://localhost:8000/models/status", timeout=2).json()
                training_mode = local_status.get("training_mode")
                training_mode_description = local_status.get("training_mode_description")
                input_dim = local_status.get("config", {}).get("input_dim")
            except Exception:
                # Don't default to any value - let the server fetch from GPU trainer
                # or display as Unknown to avoid mislabeling training mode
                pass
            
            status = {
                "gpuAvailable": self.gpu_name is not None,
                "gpuName": self.gpu_name,
                "gpuMemoryUsed": self.gpu_memory_used,
                "gpuMemoryTotal": self.gpu_memory_total,
                "gpuUtilization": self.gpu_utilization,
                "isTraining": self.is_training,
                "trainingProgress": (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0,
                "currentModel": self.current_model,
                "currentEpoch": self.current_epoch,
                "totalEpochs": self.total_epochs,
                "trainLoss": self.train_loss,
                "valLoss": self.val_loss,
                "bestValLoss": self.best_val_loss if self.best_val_loss < float('inf') else None,
                "bestEpoch": self.best_epoch,
                "modelsLoaded": self.models_completed,  # Models with loaded checkpoints
                "modelsCompleted": self.models_completed,
                "modelStatus": self.model_status,  # Per-model training status
                # Training mode detection based on feature dimensions
                "training_mode": training_mode,
                "training_mode_description": training_mode_description,
                "input_dim": input_dim
            }
            
            url = f"{proxy_url}/api/gpu/push-status"
            requests.post(url, json=status, timeout=5)
            
        except Exception:
            pass
    
    def update_model_status(self, model_name: str, status: str, accuracy: float = None, 
                            loss: float = None, epochs: int = None, best_epoch: int = None):
        """Update status for a specific model."""
        if model_name.lower() in self.model_status:
            ms = self.model_status[model_name.lower()]
            ms["status"] = status
            if accuracy is not None:
                ms["accuracy"] = accuracy
            if loss is not None:
                ms["loss"] = loss
            if epochs is not None:
                ms["epochs"] = epochs
            if best_epoch is not None:
                ms["best_epoch"] = best_epoch
    
    def update_trade_count(self, direction: str = None, reset: bool = False):
        """Update live trade count display.
        
        Args:
            direction: "long" or "short" to increment trade count
            reset: If True, reset all counts to 0
        """
        if reset:
            self.live_trade_count = {
                "total": 0, "long": 0, "short": 0, 
                "session_start": datetime.now()
            }
        elif direction:
            direction = direction.lower()
            if direction in ["long", "short"]:
                self.live_trade_count[direction] += 1
                self.live_trade_count["total"] += 1
                if self.live_trade_count["session_start"] is None:
                    self.live_trade_count["session_start"] = datetime.now()
        
        # Update UI (must be called from main thread)
        self.root.after(0, self._refresh_trade_count_display)
    
    def _refresh_trade_count_display(self):
        """Refresh the trade count UI elements (legacy - now uses OOS trade count)."""
        pass  # No longer used - OOS trade count updated via _update_oos_trade_count
    
    def _update_oos_trade_count(self, oos_trades: int, epoch: int, was_skipped: bool = False, model_name: str = None):
        """Update OOS (out-of-sample) trade count during training monitoring sweeps.
        
        Args:
            oos_trades: Number of simulated trades from monitoring sweep
            epoch: Current training epoch
            was_skipped: True if monitoring sweep was skipped this epoch
            model_name: Name of the model being trained (e.g., "transformer", "lstm")
                        If None, uses self.current_training_model
        """
        try:
            MIN_TRADES = 30  # Policy eligibility threshold (matches multihead_trainer.py)
            
            # Determine which model we're tracking
            target_model = model_name.lower() if model_name else self.current_training_model
            
            # Safeguard: log warning if no model is being tracked
            if not target_model:
                self.log(f"[OOS WARNING] No model specified for OOS trade count (epoch {epoch})")
            
            # Update per-model OOS trade count in model_status (won't be overwritten by other models)
            if target_model and target_model in self.model_status:
                if not was_skipped:
                    self.model_status[target_model]["oos_trades"] = oos_trades
                    self.model_status[target_model]["oos_epoch"] = epoch
                    self.log(f"[OOS] {target_model.upper()}: {oos_trades} trades @ epoch {epoch}")
            
            if was_skipped:
                # Monitoring sweep was skipped this epoch (runs every 5 epochs)
                # Show stale indicator but keep last known trade count
                self.trade_epoch_label.config(text=f"(sweep @epoch {epoch - (epoch % 5) if epoch % 5 != 0 else epoch})")
                return
            
            # Update UI trade count display (shows current model being trained)
            model_display = target_model.upper() if target_model else "MODEL"
            self.trade_total_label.config(text=str(oos_trades))
            
            # Update epoch indicator with model name
            self.trade_epoch_label.config(text=f"({model_display} epoch {epoch})")
            
            # Update eligibility status with color coding
            if oos_trades >= MIN_TRADES:
                self.trade_status_label.config(
                    text="✓ ELIGIBLE", 
                    fg=self.colors['success']
                )
                self.trade_total_label.config(fg=self.colors['success'])
            else:
                self.trade_status_label.config(
                    text="✗ INELIGIBLE", 
                    fg=self.colors['error']
                )
                self.trade_total_label.config(fg=self.colors['error'])
        except Exception:
            pass
            
    def start_fetch(self):
        if self.is_fetching:
            return
            
        proxy_url = self.proxy_url_var.get().strip()
        if not proxy_url:
            messagebox.showwarning("No URL", "Please enter your Replit server URL")
            return
            
        self.is_fetching = True
        self.fetch_btn.config(state=tk.DISABLED)
        self.fetch_progress['value'] = 0
        self.fetch_start_time = time.time()
        self.fetch_completed_items = 0
        
        # Only fetch 15m timeframe - matches Replit NN data and training config
        # Other timeframes (1m, 5m, 1h, 4h) are NOT used for training
        symbols = ["BTCUSDT"]  # Only BTC for focused training
        timeframes = ["15m"]   # Only 15m - matches training configuration
        self.fetch_total_items = len(symbols) * len(timeframes)
        
        def do_fetch():
            try:
                self.log(f"Starting bulk data download from Replit...")
                self.log(f"Assets: {', '.join(symbols)}")
                self.log(f"Timeframes: {', '.join(timeframes)}")
                self.log("")
                
                os.environ["REPLIT_PROXY_URL"] = proxy_url
                
                from config import Config
                config = Config()
                config.data.symbols = symbols
                config.replit_proxy_url = proxy_url
                
                from data.pipeline import BinanceDataFetcher
                
                def progress_callback(current_candles, expected_total, symbol=None, timeframe=None):
                    """
                    Callback for real-time progress during bulk download.
                    current_candles: Number of candles received so far
                    expected_total: Total candles expected (from server meta)
                    """
                    if expected_total > 0:
                        pct = (current_candles / expected_total) * 100
                    else:
                        pct = 0
                    
                    elapsed = time.time() - self.fetch_start_time
                    
                    # Calculate ETA and speed based on candles received
                    if current_candles > 0 and elapsed > 0:
                        speed = current_candles / elapsed  # candles per second
                        remaining = expected_total - current_candles
                        eta_seconds = remaining / speed if speed > 0 else -1
                    else:
                        speed = 0
                        eta_seconds = -1
                    
                    # Update UI in main thread
                    self.root.after(0, lambda p=pct, e=eta_seconds, s=speed: self._update_fetch_progress(p, e, s))
                
                fetcher = BinanceDataFetcher(
                    symbols, 
                    timeframes,
                    replit_proxy_url=proxy_url,
                    use_sync=True
                )
                
                # Use bulk download (much faster)
                self.log(f"Using bulk streaming download from Replit...")
                self.log(f"URL: {proxy_url}/api/nn-data/bulk-export")
                data = fetcher.fetch_bulk_from_replit(progress_callback=progress_callback)
                
                # Log bulk download result
                if data:
                    total_in_bulk = sum(len(df) for tfs in data.values() for df in tfs.values() if hasattr(df, '__len__'))
                    self.log(f"Bulk download returned {total_in_bulk:,} total candles")
                else:
                    self.log(f"Bulk download returned empty")
                
                # Fallback to individual fetches if bulk failed
                if not data or all(not any(len(df) > 0 for df in tfs.values()) for tfs in data.values()):
                    self.log(f"")
                    self.log(f"[WARNING] Bulk download failed!")
                    self.log(f"Falling back to individual fetches via Replit proxy...")
                    self.log(f"(This is slower but uses same data source)")
                    # Reset progress for individual fetches
                    self.root.after(0, lambda: self._update_fetch_progress(0, -1, 0))
                    data = fetcher.fetch_all_historical_sync(100000, progress_callback=progress_callback)
                
                total_candles = 0
                self.log("")
                for sym, tfs in data.items():
                    for tf, df in tfs.items():
                        if len(df) > 0:
                            path = config.data_dir / f"{sym}_{tf}.parquet"
                            df.to_parquet(path)
                            total_candles += len(df)
                            self.log(f"  Saved {len(df):,} candles: {sym} {tf}")
                        else:
                            self.log(f"  WARNING: No data for {sym} {tf}")
                            
                elapsed = time.time() - self.fetch_start_time
                if total_candles > 0:
                    self.log(f"")
                    self.log(f"Download complete!")
                    self.log(f"Total: {total_candles:,} candles in {format_time(elapsed)}")
                    self.log(f"Average speed: {total_candles / elapsed:,.0f} candles/sec")
                else:
                    self.log(f"[ERROR] No data fetched. Check connection.")
                    
            except Exception as e:
                self.log(f"[ERROR] Fetch error: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.root.after(0, self.fetch_complete)
                
        self.fetch_thread = threading.Thread(target=do_fetch, daemon=True)
        self.fetch_thread.start()
        
    def _update_fetch_progress(self, pct, eta_seconds, speed):
        self.fetch_progress['value'] = pct
        self.fetch_pct_label.config(text=f"{pct:.0f}%")
        
        if eta_seconds >= 0:
            self.fetch_eta_label.config(text=f"ETA: {format_time(eta_seconds)}")
        else:
            self.fetch_eta_label.config(text="")
            
        if speed > 0:
            self.fetch_speed_label.config(text=f"{speed:,.0f} c/s")
        
    def fetch_complete(self):
        self.is_fetching = False
        self.fetch_btn.config(state=tk.NORMAL)
        self.fetch_progress['value'] = 100
        self.fetch_pct_label.config(text="100%")
        self.fetch_eta_label.config(text="Complete")
        
    def start_training(self):
        if self.is_training:
            return
        
        model_type = self.model_var.get()
        defaults = OPTIMAL_DEFAULTS.get(model_type, OPTIMAL_DEFAULTS["transformer"])
        
        # Get selected assets from GUI
        training_assets = self.get_selected_assets()
        if not training_assets:
            messagebox.showwarning("No Assets", "Please select at least one asset to train on.")
            return
        
        # Get timeframe mode from GUI
        tf_mode = self.get_timeframe_mode()
        
        # MTF Fusion Mode: Base timeframe is 15m, with 5m/1h/4h as context
        # All predictions are for 15m timeframe with 2-3 hour horizon (10 bars)
        if tf_mode == "15m":
            # Quick 15m mode - uses only 15m timeframe data
            use_mtf_fusion = False
            mtf_base_tf = "15m"
            mtf_context_tfs = []
            mtf_all_tfs = ["15m"]  # Only 15m data
            training_timeframes = ["15m"]
        else:
            # Full MTF mode - uses 5m, 15m, 1h, 4h with feature fusion
            use_mtf_fusion = True
            mtf_base_tf = "15m"
            mtf_context_tfs = ["5m", "1h", "4h"]
            mtf_all_tfs = ["5m", "15m", "1h", "4h"]
            training_timeframes = mtf_all_tfs
        
        prediction_horizon_bars = 10  # 10 x 15m = 2.5 hours forward
        
        # Legacy mode: treat each timeframe as separate samples
        horizon_by_tf = {"1m": 240, "5m": 48, "15m": 16, "1h": 4, "4h": 1}
        
        # Check for training data - need at least BTC for some timeframe
        data_dir = Path(__file__).parent / "data_cache"
        has_data = any((data_dir / f"BTCUSDT_{tf}.parquet").exists() for tf in training_timeframes)
        
        if not has_data:
            result = messagebox.askyesno(
                "No Data", 
                f"No training data found.\nDownload data first?"
            )
            if result:
                self.start_fetch()
            return
            
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)
        self.train_progress['value'] = 0
        self.training_start_time = time.time()
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Track which model is being trained (for per-model OOS trade tracking)
        self.current_training_model = model_type.lower()
        
        epochs = defaults["epochs"]
        batch_size = defaults["batch_size"]
        lr = defaults["lr"]
        trading_cost = self.get_trading_cost()  # Capture selected cost mode
        cost_mode = self.cost_mode_var.get()
        use_multihead = self.multihead_var.get()  # Capture multi-head mode selection
        
        def do_train():
            # Initialize feature_cols at function scope (set in MTF or legacy branch)
            feature_cols = None
            
            try:
                self.log(f"")
                self.log(f"{'='*55}")
                if use_mtf_fusion:
                    self.log(f"  MTF FUSION TRAINING: {model_type.upper()}")
                    self.log(f"  Base: 15m | Context: 5m, 1h, 4h | Horizon: {prediction_horizon_bars} bars (2.5h)")
                else:
                    self.log(f"  QUICK 15M TRAINING: {model_type.upper()}")
                    self.log(f"  Timeframe: 15m only | Horizon: {prediction_horizon_bars} bars (2.5h)")
                    self.log(f"  Mode: Fast training with ~57 features")
                self.log(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
                self.log(f"  Cost: {cost_mode} ({trading_cost*100:.2f}%)")
                self.log(f"  Assets: {', '.join(training_assets)}")
                self.log(f"{'='*55}")
                self.log(f"")
                
                import torch
                import numpy as np
                import pandas as pd
                from config import config
                from data.pipeline import FeatureEngineer, TradingDataset, create_labels
                from torch.utils.data import DataLoader
                
                # Import appropriate trainer based on mode
                if use_multihead:
                    from training.multihead_trainer import MultiHeadTrainer, MultiHeadDataset, MultiHeadLossConfig
                    from data.regression_targets import generate_multihead_targets
                    self.log(f"  Mode: MULTI-HEAD (Classification + Regression + Quantile)")
                else:
                    from training.trainer import Trainer
                    self.log(f"  Mode: LEGACY (Classification only)")
                
                if use_mtf_fusion:
                    # === MTF FUSION MODE ===
                    # Load all timeframes for each asset and fuse to 15m base
                    from data.mtf_fusion import MTFFeatureFusion, add_cross_asset_features
                    
                    self.log(f"Loading multi-timeframe data for MTF fusion...")
                    
                    asset_data = {}  # {symbol: {tf: df}}
                    total_candles = 0
                    
                    for asset in training_assets:
                        tf_data = {}
                        for tf in mtf_all_tfs:
                            asset_path = data_dir / f"{asset}_{tf}.parquet"
                            if asset_path.exists():
                                df = pd.read_parquet(asset_path)
                                tf_data[tf] = df
                                total_candles += len(df)
                                self.log(f"  {asset} {tf}: {len(df):,} candles")
                            time.sleep(0)
                        if tf_data:
                            asset_data[asset] = tf_data
                    
                    if not asset_data:
                        self.log("ERROR: No data files found!")
                        self.root.after(0, self.training_complete)
                        return
                    
                    self.log(f"")
                    self.log(f"Total: {total_candles:,} raw candles")
                    self.log(f"")
                    self.log(f"Fusing timeframes to 15m base (leakage-proof alignment)...")
                    
                    # Fuse all timeframes for each asset
                    fusioner = MTFFeatureFusion(prediction_horizon_bars)
                    all_fused = []
                    
                    for symbol, tf_data in asset_data.items():
                        if mtf_base_tf not in tf_data:
                            self.log(f"  {symbol}: Skipping - no 15m data")
                            continue
                        fused = fusioner.align_timeframes(tf_data, symbol)
                        fused["symbol"] = symbol
                        all_fused.append(fused)
                        self.log(f"  {symbol}: {len(fused):,} fused samples, {len(fused.columns)} features")
                        time.sleep(0)
                    
                    if not all_fused:
                        self.log("ERROR: No fused data!")
                        self.root.after(0, self.training_complete)
                        return
                    
                    # Combine all assets
                    combined = pd.concat(all_fused, ignore_index=True)
                    self.log(f"")
                    self.log(f"Combined: {len(combined):,} samples")
                    
                    # NOTE: Cross-asset features REMOVED (FIX #1 - training-inference alignment)
                    # These features cannot be computed at inference with BTC-only data
                    self.log(f"Skipping cross-asset features (BTC-only inference alignment)")
                    
                    # Time-based train/val split per asset
                    self.log(f"Splitting by time per asset...")
                    train_ratio = 0.70
                    val_ratio = 0.15
                    
                    train_dfs = []
                    val_dfs = []
                    
                    for symbol in combined["symbol"].unique():
                        asset_df = combined[combined["symbol"] == symbol].copy()
                        asset_df = asset_df.sort_values("datetime").reset_index(drop=True)
                        n = len(asset_df)
                        n_train = int(n * train_ratio)
                        n_val = int(n * val_ratio)
                        
                        # Split and add purge gap (prediction_horizon_bars)
                        train_df = asset_df.iloc[:n_train - prediction_horizon_bars].copy()
                        val_df = asset_df.iloc[n_train:n_train + n_val].copy()
                        
                        train_dfs.append(train_df)
                        val_dfs.append(val_df)
                        self.log(f"  {symbol}: train={len(train_df):,}, val={len(val_df):,}")
                    
                    train_combined = pd.concat(train_dfs, ignore_index=True)
                    val_combined = pd.concat(val_dfs, ignore_index=True)
                    
                    # Create labels
                    self.log(f"Creating labels (horizon={prediction_horizon_bars} bars, ~2.5h)...")
                    
                    if use_multihead:
                        # Multi-head mode: generate class_labels and forward_returns
                        # Label mode selection (HOLD fix):
                        #   - "cost_aware" (Stage 1): min_confidence gating
                        #   - "pure_directional" (Stage 2): simple return threshold
                        #   - "regime" (Stage 3): ADX-based adaptive thresholds
                        label_mode = getattr(self, 'label_mode', 'regime')
                        use_pure_directional = (label_mode == "pure_directional")
                        use_regime_labels = (label_mode == "regime")
                        directional_threshold = getattr(self, 'directional_threshold', 0.0020)
                        trend_threshold = getattr(self, 'trend_threshold', 0.0015)
                        range_threshold = getattr(self, 'range_threshold', 0.0030)
                        min_confidence = getattr(self, 'min_confidence', 0.40)
                        
                        self.log(f"  Label mode: {label_mode.upper()}")
                        if use_regime_labels:
                            self.log(f"  Using REGIME mode (trend={trend_threshold:.4%}, range={range_threshold:.4%})")
                        elif use_pure_directional:
                            self.log(f"  Using PURE DIRECTIONAL mode (threshold={directional_threshold:.4%})")
                        else:
                            self.log(f"  Using COST-AWARE mode (min_confidence={min_confidence})")
                        
                        train_targets = generate_multihead_targets(
                            train_combined, 
                            horizon_periods=prediction_horizon_bars,
                            min_net_edge=0.0,  # No edge filter for debugging
                            min_confidence=min_confidence,  # Stage 1
                            use_volatility_cost=False,
                            fixed_cost=0.0009,  # 0.09% taker/taker
                            use_pure_directional=use_pure_directional,  # Stage 2
                            directional_threshold=directional_threshold,
                            use_regime_labels=use_regime_labels,  # Stage 3
                            trend_threshold=trend_threshold,
                            range_threshold=range_threshold
                        )
                        val_targets = generate_multihead_targets(
                            val_combined, 
                            horizon_periods=prediction_horizon_bars,
                            min_net_edge=0.0,
                            min_confidence=min_confidence,  # Stage 1
                            use_volatility_cost=False,
                            fixed_cost=0.0009,
                            use_pure_directional=use_pure_directional,  # Stage 2
                            directional_threshold=directional_threshold,
                            use_regime_labels=use_regime_labels,  # Stage 3
                            trend_threshold=trend_threshold,
                            range_threshold=range_threshold
                        )
                        
                        train_labels = train_targets['class_label']
                        val_labels = val_targets['class_label']
                        train_returns = train_targets['forward_return']
                        val_returns = val_targets['forward_return']
                        self.log(f"  Generated multi-head targets with forward returns")
                    else:
                        train_labels = fusioner.create_labels(train_combined)
                        val_labels = fusioner.create_labels(val_combined)
                        train_returns = None
                        val_returns = None
                    
                    # Drop rows with NaN labels (end of each asset's data)
                    train_valid = train_labels.notna()
                    val_valid = val_labels.notna()
                    train_combined = train_combined[train_valid].reset_index(drop=True)
                    train_labels = train_labels[train_valid].reset_index(drop=True)
                    val_combined = val_combined[val_valid].reset_index(drop=True)
                    val_labels = val_labels[val_valid].reset_index(drop=True)
                    
                    if use_multihead:
                        train_returns = train_returns[train_valid].reset_index(drop=True)
                        val_returns = val_returns[val_valid].reset_index(drop=True)
                    
                    # Select numeric feature columns only
                    exclude_cols = ["datetime", "symbol", "timestamp", "open", "high", "low", "close", "volume"]
                    feature_cols = [c for c in train_combined.columns if c not in exclude_cols and train_combined[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
                    
                    train_features_raw = train_combined[feature_cols].copy()
                    val_features_raw = val_combined[feature_cols].copy()
                    train_labels = train_labels.values.astype(np.int64)
                    val_labels = val_labels.values.astype(np.int64)
                    
                    self.log(f"")
                    self.log(f"MTF features: {len(feature_cols)} columns")
                    self.log(f"Train: {len(train_features_raw):,}, Val: {len(val_features_raw):,}")
                    
                    # Fit scalers on training data only
                    engineer = FeatureEngineer()
                    
                else:
                    # === LEGACY MODE: Each timeframe as separate samples ===
                    all_dfs = []
                    all_horizons = []
                    total_candles = 0
                    
                    for asset in training_assets:
                        asset_candles = 0
                        for tf in training_timeframes:
                            asset_path = data_dir / f"{asset}_{tf}.parquet"
                            if asset_path.exists():
                                df = pd.read_parquet(asset_path)
                                df['symbol'] = asset
                                df['timeframe'] = tf
                                all_dfs.append(df)
                                all_horizons.append(horizon_by_tf.get(tf, 16))
                                asset_candles += len(df)
                                self.log(f"  {asset} {tf}: {len(df):,} candles")
                            time.sleep(0)
                        if asset_candles > 0:
                            total_candles += asset_candles
                    
                    self.log(f"")
                    self.log(f"Total: {total_candles:,} candles across {len(all_dfs)} asset-timeframe pairs")
                    
                    if not all_dfs:
                        self.log("ERROR: No data files found!")
                        self.root.after(0, self.training_complete)
                        return
                    
                    self.log(f"")
                    self.log(f"Processing features per-asset-timeframe with time-based splits...")
                    
                    train_ratio = 0.70
                    val_ratio = 0.15
                    
                    train_features_list = []
                    train_labels_list = []
                    val_features_list = []
                    val_labels_list = []
                    train_returns_list = []
                    val_returns_list = []
                    engineer = FeatureEngineer()
                    
                    for i, df in enumerate(all_dfs):
                        asset = df['symbol'].iloc[0] if 'symbol' in df.columns else f"asset_{i}"
                        tf = df['timeframe'].iloc[0] if 'timeframe' in df.columns else "15m"
                        horizon = all_horizons[i]
                        
                        n = len(df)
                        n_train = int(n * train_ratio)
                        n_val = int(n * val_ratio)
                        
                        train_df = df.iloc[:n_train].copy()
                        val_df = df.iloc[n_train:n_train + n_val].copy()
                        
                        train_features = engineer.compute_technical_features(train_df).fillna(0)
                        val_features = engineer.compute_technical_features(val_df).fillna(0)
                        
                        if use_multihead:
                            # Use same label_mode settings as STF path
                            label_mode = getattr(self, 'label_mode', 'regime')
                            use_pure_directional = (label_mode == "pure_directional")
                            use_regime_labels = (label_mode == "regime")
                            directional_threshold = getattr(self, 'directional_threshold', 0.0020)
                            trend_threshold = getattr(self, 'trend_threshold', 0.0015)
                            range_threshold = getattr(self, 'range_threshold', 0.0030)
                            min_confidence = getattr(self, 'min_confidence', 0.40)
                            
                            train_targets = generate_multihead_targets(
                                train_df, 
                                horizon_periods=horizon,
                                min_net_edge=0.0,
                                min_confidence=min_confidence,  # Stage 1: 0.40
                                use_volatility_cost=False,
                                fixed_cost=0.0009,
                                use_pure_directional=use_pure_directional,  # Stage 2
                                directional_threshold=directional_threshold,
                                use_regime_labels=use_regime_labels,  # Stage 3
                                trend_threshold=trend_threshold,
                                range_threshold=range_threshold
                            )
                            val_targets = generate_multihead_targets(
                                val_df, 
                                horizon_periods=horizon,
                                min_net_edge=0.0,
                                min_confidence=min_confidence,  # Stage 1: 0.40
                                use_volatility_cost=False,
                                fixed_cost=0.0009,
                                use_pure_directional=use_pure_directional,  # Stage 2
                                directional_threshold=directional_threshold,
                                use_regime_labels=use_regime_labels,  # Stage 3
                                trend_threshold=trend_threshold,
                                range_threshold=range_threshold
                            )
                            
                            train_labels_arr = train_targets['class_label'].values
                            val_labels_arr = val_targets['class_label'].values
                            train_returns_arr = train_targets['forward_return'].values
                            val_returns_arr = val_targets['forward_return'].values
                            
                            # CRITICAL: Align features and targets to same length BEFORE filtering
                            # Features and targets may have different lengths due to horizon offset
                            min_train_len = min(len(train_features), len(train_labels_arr))
                            min_val_len = min(len(val_features), len(val_labels_arr))
                            
                            train_features = train_features.iloc[:min_train_len]
                            train_labels_arr = train_labels_arr[:min_train_len]
                            train_returns_arr = train_returns_arr[:min_train_len]
                            
                            val_features = val_features.iloc[:min_val_len]
                            val_labels_arr = val_labels_arr[:min_val_len]
                            val_returns_arr = val_returns_arr[:min_val_len]
                            
                            # Now apply NaN filtering with aligned arrays
                            valid_train = ~np.isnan(train_labels_arr)
                            valid_val = ~np.isnan(val_labels_arr)
                            
                            train_features = train_features[valid_train]
                            train_labels_arr = train_labels_arr[valid_train].astype(int)
                            train_returns_arr = train_returns_arr[valid_train]
                            
                            val_features = val_features[valid_val]
                            val_labels_arr = val_labels_arr[valid_val].astype(int)
                            val_returns_arr = val_returns_arr[valid_val]
                            
                            train_returns_list.append(train_returns_arr)
                            val_returns_list.append(val_returns_arr)
                        else:
                            train_labels_arr = create_labels(train_df, horizon=horizon, threshold=0.001, trading_cost=trading_cost)
                            val_labels_arr = create_labels(val_df, horizon=horizon, threshold=0.001, trading_cost=trading_cost)
                            
                            train_labels_arr = (train_labels_arr + 1).astype(int)
                            val_labels_arr = (val_labels_arr + 1).astype(int)
                            
                            if len(train_features) > horizon:
                                train_features = train_features.iloc[:-horizon]
                                train_labels_arr = train_labels_arr[:-horizon]
                        
                        train_features_list.append(train_features)
                        train_labels_list.append(train_labels_arr)
                        val_features_list.append(val_features)
                        val_labels_list.append(val_labels_arr)
                        
                        self.log(f"  {asset} {tf} (h={horizon}): train={len(train_features):,}, val={len(val_features):,}")
                        time.sleep(0)
                    
                    train_features_raw = pd.concat(train_features_list, ignore_index=True)
                    val_features_raw = pd.concat(val_features_list, ignore_index=True)
                    train_labels = np.concatenate(train_labels_list)
                    val_labels = np.concatenate(val_labels_list)
                    
                    if use_multihead:
                        train_returns = np.concatenate(train_returns_list)
                        val_returns = np.concatenate(val_returns_list)
                        self.log(f"  Generated multi-head targets with forward returns")
                    else:
                        train_returns = None
                        val_returns = None
                
                self.log(f"")
                self.log(f"Combined: train={len(train_features_raw):,}, val={len(val_features_raw):,}")
                
                # Store feature columns for later saving (both MTF and legacy modes)
                # In MTF mode, feature_cols was set at line ~1380; in legacy mode, set here
                if feature_cols is None:
                    feature_cols = list(train_features_raw.columns)
                
                # === FIT SCALERS ON TRAINING DATA ONLY ===
                self.log(f"Fitting scalers on training data only (no leakage)")
                engineer.fit_scalers(train_features_raw)
                
                # Transform both sets using training-fitted scalers
                train_features_scaled = engineer.transform(train_features_raw)
                val_features_scaled = engineer.transform(val_features_raw)
                
                train_features_np = train_features_scaled.values.astype(np.float32)
                val_features_np = val_features_scaled.values.astype(np.float32)
                train_labels_np = train_labels.astype(np.int64)
                val_labels_np = val_labels.astype(np.int64)
                
                # Skip initial sequence_length samples
                valid_start = config.data.sequence_length
                train_features_np = train_features_np[valid_start:]
                train_labels_np = train_labels_np[valid_start:]
                val_features_np = val_features_np[valid_start:]
                val_labels_np = val_labels_np[valid_start:]
                
                # CRITICAL: Also skip valid_start from returns to keep arrays aligned
                if use_multihead and train_returns is not None:
                    train_returns = train_returns[valid_start:]
                    val_returns = val_returns[valid_start:]
                
                # === HARD DATA CLEANSING - Drop NaN/Inf rows ===
                def clean_data_gui(features, labels, returns=None, name="Data"):
                    features = np.where(np.isinf(features), np.nan, features)
                    nan_mask = np.isnan(features).any(axis=1)
                    if returns is not None:
                        returns = np.where(np.isinf(returns), np.nan, returns)
                        nan_mask = nan_mask | np.isnan(returns)
                    nan_count = nan_mask.sum()
                    if nan_count > 0:
                        self.log(f"{name}: Dropping {nan_count} rows with NaN/Inf ({nan_count/len(features)*100:.1f}%)")
                        valid_mask = ~nan_mask
                        features = features[valid_mask]
                        labels = labels[valid_mask]
                        if returns is not None:
                            returns = returns[valid_mask]
                    assert np.isfinite(features).all(), f"{name}: Non-finite values remain!"
                    self.log(f"{name}: {len(features)} clean samples")
                    if returns is not None:
                        return features, labels, returns
                    return features, labels
                
                if use_multihead:
                    train_returns_np = train_returns.values.astype(np.float32) if hasattr(train_returns, 'values') else train_returns.astype(np.float32)
                    val_returns_np = val_returns.values.astype(np.float32) if hasattr(val_returns, 'values') else val_returns.astype(np.float32)
                    train_features_np, train_labels_np, train_returns_np = clean_data_gui(
                        train_features_np, train_labels_np, train_returns_np, name="Train")
                    val_features_np, val_labels_np, val_returns_np = clean_data_gui(
                        val_features_np, val_labels_np, val_returns_np, name="Val")
                else:
                    train_features_np, train_labels_np = clean_data_gui(train_features_np, train_labels_np, name="Train")
                    val_features_np, val_labels_np = clean_data_gui(val_features_np, val_labels_np, name="Val")
                    train_returns_np = None
                    val_returns_np = None
                
                # === CLASS WEIGHT BALANCING ===
                # Cap weights to prevent gradient explosion (max 10x)
                MAX_CLASS_WEIGHT = 10.0
                class_counts = np.bincount(train_labels_np, minlength=3)
                total_samples = len(train_labels_np)
                class_weights = total_samples / (3 * class_counts + 1e-6)
                class_weights = np.clip(class_weights, 1.0, MAX_CLASS_WEIGHT)  # Cap to prevent NaN
                class_weights_tensor = torch.FloatTensor(class_weights)
                
                self.log(f"Class distribution: SHORT={class_counts[0]:,}, NEUTRAL={class_counts[1]:,}, LONG={class_counts[2]:,}")
                self.log(f"Class weights (capped at {MAX_CLASS_WEIGHT}x): [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]")
                
                # Create datasets (with data validation)
                if use_multihead:
                    train_dataset = MultiHeadDataset(train_features_np, train_labels_np, train_returns_np, config.data.sequence_length)
                    val_dataset = MultiHeadDataset(val_features_np, val_labels_np, val_returns_np, config.data.sequence_length)
                    self.log(f"Created MultiHeadDataset (features, labels, returns)")
                else:
                    train_dataset = TradingDataset(train_features_np, train_labels_np, config.data.sequence_length, validate_data=True)
                    val_dataset = TradingDataset(val_features_np, val_labels_np, config.data.sequence_length, validate_data=True)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                input_dim = train_features_np.shape[1]
                self.log(f"")
                self.log(f"Features: {input_dim}, Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
                self.log(f"")
                
                # Model creation
                if use_multihead:
                    # Multi-head models with Classification + Regression + Quantile heads
                    from models.multihead import MultiHeadTransformer, MultiHeadTFT, MultiHeadLSTM, MultiHeadCNN, MultiHeadGNN, MultiHeadVAE
                    
                    if model_type == "transformer":
                        model = MultiHeadTransformer(input_dim=input_dim, d_model=256, nhead=8, num_layers=6)
                    elif model_type == "tft":
                        model = MultiHeadTFT(input_dim=input_dim, d_model=256, nhead=8, num_encoder_layers=4)
                    elif model_type == "lstm":
                        model = MultiHeadLSTM(input_dim=input_dim, hidden_dim=256, num_layers=3)
                    elif model_type == "cnn":
                        model = MultiHeadCNN(input_dim=input_dim, hidden_channels=256, num_blocks=4)
                    elif model_type == "gnn":
                        model = MultiHeadGNN(input_dim=input_dim, hidden_dim=128, num_layers=3)
                    elif model_type == "vae":
                        model = MultiHeadVAE(input_dim=input_dim, sequence_length=config.data.sequence_length, latent_dim=64)
                    else:
                        self.log(f"Multi-head mode not supported for: {model_type}")
                        self.log(f"Supported: transformer, tft, lstm, cnn, gnn, vae")
                        self.root.after(0, self.training_complete)
                        return
                    self.log(f"Multi-head model: {model.name}")
                else:
                    # Legacy classification-only models
                    if model_type == "transformer":
                        from models.transformer import TransformerPriceModel
                        model = TransformerPriceModel(input_dim=input_dim, d_model=256, nhead=8, num_layers=6)
                    elif model_type == "tft":
                        from models.transformer import TemporalFusionTransformer
                        model = TemporalFusionTransformer(input_dim=input_dim, d_model=256, nhead=8)
                    elif model_type == "lstm":
                        from models.lstm import BidirectionalLSTM
                        model = BidirectionalLSTM(input_dim=input_dim, hidden_dim=256, num_layers=3)
                    elif model_type == "cnn":
                        from models.cnn import ResNetPrice
                        model = ResNetPrice(input_dim=input_dim, channels=[64, 128, 256, 512])
                    elif model_type == "vae":
                        from models.vae import MarketVAE
                        model = MarketVAE(input_dim=input_dim, sequence_length=config.data.sequence_length, latent_dim=64)
                    elif model_type == "gnn":
                        from models.gnn import CrossAssetGNN
                        model = CrossAssetGNN(input_dim=input_dim, num_assets=4)
                    else:
                        self.log(f"Unknown model: {model_type}")
                        return
                    
                self.log(f"Parameters: {model.count_parameters():,}")
                
                config.training.epochs = epochs
                config.training.learning_rate = lr
                
                # Create trainer
                if use_multihead:
                    loss_config = MultiHeadLossConfig(class_weights=class_weights_tensor)
                    trainer = MultiHeadTrainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=config,
                        device=config.device,
                        loss_config=loss_config,
                        gui_mode=True
                    )
                    self.log(f"Using MultiHeadTrainer with combined loss:")
                    self.log(f"  CrossEntropy + Huber + GaussianNLL + Pinball")
                else:
                    trainer = Trainer(model, train_loader, val_loader, config, device=config.device, 
                                      gui_mode=True, class_weights=class_weights_tensor)
                
                self.current_model = model_type
                self.total_epochs = epochs
                
                def progress_callback(epoch, total_epochs, train_metrics, val_metrics):
                    # Extract losses from metrics dictionaries (MultiHeadTrainer format)
                    if isinstance(train_metrics, dict):
                        train_loss = train_metrics.get('total', train_metrics.get('total_loss', train_metrics.get('loss', 0.0)))
                    else:
                        train_loss = float(train_metrics)
                    if isinstance(val_metrics, dict):
                        val_loss = val_metrics.get('total', val_metrics.get('total_loss', val_metrics.get('loss', 0.0)))
                        # Only update OOS trade count when monitoring sweep actually ran
                        # (num_trades key exists in val_metrics - sweeps run every 5 epochs)
                        if 'num_trades' in val_metrics:
                            oos_trades = val_metrics['num_trades']
                            was_skipped = val_metrics.get('_skipped', False)
                            # Capture current_training_model at callback time to ensure per-model tracking
                            current_model = self.current_training_model
                            self.root.after(0, lambda t=oos_trades, e=epoch+1, skip=was_skipped, m=current_model: 
                                           self._update_oos_trade_count(t, e, skip, m))
                    else:
                        val_loss = float(val_metrics)
                    
                    epoch_end_time = time.time()
                    if len(self.epoch_times) > 0:
                        epoch_duration = epoch_end_time - self.epoch_times[-1]
                    else:
                        epoch_duration = epoch_end_time - self.training_start_time
                    self.epoch_times.append(epoch_end_time)
                    
                    # Calculate ETA
                    epochs_remaining = total_epochs - (epoch + 1)
                    if len(self.epoch_times) >= 2:
                        avg_epoch_time = (self.epoch_times[-1] - self.training_start_time) / (epoch + 1)
                        eta_seconds = avg_epoch_time * epochs_remaining
                    else:
                        eta_seconds = epoch_duration * epochs_remaining
                    
                    # Track best
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch + 1
                    
                    self.current_epoch = epoch + 1
                    self.train_loss = train_loss
                    self.val_loss = val_loss
                    
                    progress = (epoch + 1) / total_epochs * 100
                    
                    self.root.after(0, lambda: self._update_training_progress(
                        progress, epoch + 1, total_epochs, train_loss, val_loss, eta_seconds
                    ))
                    
                    self.log(f"Epoch {epoch+1:3d}/{total_epochs}: loss={train_loss:.4f}, val={val_loss:.4f}" + 
                             (f" ★ best" if val_loss == self.best_val_loss else ""))
                    
                    return self.is_training
                    
                trainer.epoch_callback = progress_callback
                
                # Mark model as training
                self.update_model_status(model_type, "training")
                
                history = trainer.train(num_epochs=epochs)
                
                if self.is_training:
                    self.models_completed.append(model_type)
                    
                    # Save to checkpoints/ directory with best_* pattern (matches API server expectations)
                    checkpoint_dir = Path(__file__).parent / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_suffix = "_multihead" if use_multihead else ""
                    save_path = checkpoint_dir / f"best_{model_type}{model_suffix}.pt"
                    model.save(str(save_path))
                    
                    elapsed = time.time() - self.training_start_time
                    self.log(f"")
                    self.log(f"Training complete in {format_time(elapsed)}")
                    self.log(f"Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
                    self.log(f"Model saved: {save_path.name}")
                    
                    # Save scalers to checkpoints/ directory with model-specific naming
                    scaler_filename = f"scaler_{model_type}{model_suffix}.joblib"
                    engineer.save_scalers(str(checkpoint_dir / scaler_filename))
                    self.log(f"Scalers saved: {scaler_filename}")
                    
                    # Save feature columns for inference alignment
                    if feature_cols is not None and len(feature_cols) > 0:
                        feature_columns_path = checkpoint_dir / f"feature_columns_{model_type}{model_suffix}.txt"
                        with open(feature_columns_path, 'w') as f:
                            for col in feature_cols:
                                f.write(f"{col}\n")
                        self.log(f"Feature columns saved: {feature_columns_path.name} ({len(feature_cols)} features)")
                    else:
                        self.log(f"WARNING: feature_cols not available, skipping feature_columns.txt save")
                    
                    # Also save a generic feature_columns.txt for API server
                    if feature_cols is not None and len(feature_cols) > 0:
                        generic_feature_path = checkpoint_dir / "feature_columns.txt"
                        with open(generic_feature_path, 'w') as f:
                            for col in feature_cols:
                                f.write(f"{col}\n")
                    
                    # Update model status as complete
                    self.update_model_status(
                        model_type, "complete",
                        loss=self.best_val_loss,
                        epochs=self.current_epoch,
                        best_epoch=self.best_epoch
                    )
                else:
                    self.log("Training stopped by user")
                    self.update_model_status(model_type, "stopped")
                    
            except Exception as e:
                self.log(f"[ERROR] Training error: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.root.after(0, self.training_complete)
                
        self.training_thread = threading.Thread(target=do_train, daemon=True)
        self.training_thread.start()
        
    def _update_training_progress(self, progress, epoch, total, train_loss, val_loss, eta_seconds):
        self.train_progress['value'] = progress
        self.train_pct_label.config(text=f"Epoch {epoch}/{total}")
        self.train_eta_label.config(text=f"ETA: {format_time(eta_seconds)}")
        self.train_loss_label.config(text=f"Train: {train_loss:.4f}")
        self.val_loss_label.config(text=f"Val: {val_loss:.4f}")
        self.best_label.config(text=f"Best: {self.best_val_loss:.4f} (E{self.best_epoch})")
        
    def training_complete(self):
        self.is_training = False
        self.current_training_model = None  # Clear current training model
        self.train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.train_pct_label.config(text="Ready")
        self.train_eta_label.config(text="")
        
    def stop_training(self):
        if self.is_training:
            self.log("Stopping training (finishing current epoch)...")
            self.is_training = False
            # Also stop the train-all sequence if running
            if hasattr(self, '_train_all_stopped'):
                self._train_all_stopped = True
            
    def train_all_models(self):
        if self.is_training:
            messagebox.showinfo("Busy", "Training already in progress")
            return
            
        result = messagebox.askyesno(
            "Train All Models",
            "Train all 6 model architectures with optimal settings?\n\nThis may take several hours."
        )
        if not result:
            return
            
        def train_sequence():
            models = list(OPTIMAL_DEFAULTS.keys())
            self._train_all_stopped = False  # Flag to allow manual stop
            
            for i, model in enumerate(models):
                # Check if user manually stopped the sequence
                if self._train_all_stopped:
                    self.log(f"Training sequence stopped by user")
                    break
                    
                self.log(f"")
                self.log(f"=== Model {i+1}/{len(models)}: {model.upper()} ===")
                self.root.after(0, lambda m=model: self.model_var.set(m))
                self.root.after(100, self.start_training)
                
                # Wait for training to start (with timeout)
                start_wait = 0
                while not self.is_training and start_wait < 10:
                    time.sleep(0.5)
                    start_wait += 1
                    
                if not self.is_training:
                    self.log(f"WARNING: Training failed to start for {model}, skipping...")
                    continue
                
                # Wait for training to complete (is_training becomes False when done)
                while self.is_training:
                    time.sleep(1)
                    
                # Small delay between models
                time.sleep(1)
                    
            self.log("")
            self.log("=== All models complete! ===")
            
        threading.Thread(target=train_sequence, daemon=True).start()
        
    def start_api_server(self):
        self.log("Starting prediction API server on port 8000...")
        
        def do_serve():
            try:
                from api.server import start_server
                start_server(host="0.0.0.0", port=8000)
            except Exception as e:
                self.log(f"[ERROR] Server error: {e}")
                
        threading.Thread(target=do_serve, daemon=True).start()
        self.log("API available at http://localhost:8000")

def main():
    root = tk.Tk()
    
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
        
    app = GPUTrainerGUI(root)
    
    def on_closing():
        sys.stdout = app.original_stdout
        sys.stderr = app.original_stderr
        
        if app.is_training or app.is_fetching:
            if messagebox.askokcancel("Quit", "Operation in progress. Quit anyway?"):
                app.is_training = False
                app.is_fetching = False
                root.destroy()
        else:
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
