"""
Simple MLP Classifier for Stable Training

This is a baseline model designed for stable gradient behavior.
The complex LSTM/Transformer architectures were causing gradient explosions
even on clean data. This simple MLP trains stably with gradient norms < 1.

Use this as a baseline to verify training pipeline works before adding complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import math

# Import shared MultiHeadOutput from multihead.py for compatibility
from .multihead import MultiHeadOutput


@dataclass
class SimpleMLP_Config:
    """Configuration for SimpleMLP model."""
    input_dim: int = 41  # Number of features
    hidden_dims: list = None  # Hidden layer dimensions
    num_classes: int = 3  # SHORT, HOLD, LONG
    dropout: float = 0.3  # Dropout rate
    use_layer_norm: bool = True  # Use LayerNorm for stability
    n_candle_steps: int = 5  # For dummy candle output shape
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class SimpleMLP(nn.Module):
    """
    Simple MLP classifier designed for stable training.
    
    Key stability features:
    1. LayerNorm after each layer (prevents internal covariate shift)
    2. Orthogonal initialization (better gradient flow)
    3. GELU activation (smoother than ReLU)
    4. Moderate dropout (regularization without gradient issues)
    5. Residual connections in wider layers
    
    This model trains stably with gradient norms < 1 on the same data
    that causes LSTM/Transformer to explode to 30+.
    """
    
    def __init__(self, config: SimpleMLP_Config):
        super().__init__()
        self.config = config
        
        # Store attributes for trainer compatibility (checkpoint saving)
        self.input_dim = config.input_dim
        self.output_dim = config.num_classes  # 3 for SHORT/HOLD/LONG
        self.hidden_dims = config.hidden_dims  # For checkpoint config
        
        # Training metadata (required by base model interface)
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, config.num_classes)
        
        # Store dimensions for dummy outputs
        self.n_candle_steps = config.n_candle_steps
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits.
        
        Args:
            x: [batch, seq_len, features] or [batch, features]
            
        Returns:
            class_logits: [batch, 3]
        """
        # Handle sequence input - take last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch, features]
        
        # Forward through trunk
        features = self.trunk(x)
        
        # Classification logits
        logits = self.classifier(features)
        
        return logits
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """
        Multi-head forward pass matching the MultiHeadTransformer interface.
        
        Only classification head is active. All other heads return None to
        signal they should be skipped in loss computation.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Get classification logits
        class_logits = self.forward(x)
        
        # Required fields: class_logits and mu/quantiles (non-optional in MultiHeadOutput)
        # Set required fields to zeros, optional fields to None
        zeros_1 = torch.zeros(batch_size, 1, device=device)
        zeros_5 = torch.zeros(batch_size, 5, device=device)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=zeros_1,  # Required field
            quantiles=zeros_5,  # Required field
            # All optional fields set to None to skip in loss computation
            sigma=None,
            entry_offset=None,
            sl_distance=None,
            tp_distance=None,
            candle_deltas=None,
            vol_state_logits=None,
            acceleration=None
        )
    
    def parameters_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model checkpoint to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'created_at': self.created_at,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epochs_trained
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: str = 'cuda'):
        """Load model checkpoint from file."""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_trained = checkpoint.get('epochs_trained', 0)


def create_simple_mlp(input_dim: int = 41, num_classes: int = 3) -> SimpleMLP:
    """Factory function to create a SimpleMLP with default config."""
    config = SimpleMLP_Config(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_classes=num_classes,
        dropout=0.3,
        use_layer_norm=True
    )
    return SimpleMLP(config)


# =============================================================================
# MultiHeadSimpleMLP: Progressive complexity re-addition
# =============================================================================

@dataclass  
class MultiHeadSimpleMLP_Config:
    """Configuration for MultiHeadSimpleMLP with progressive head enablement."""
    input_dim: int = 41
    hidden_dims: list = None
    num_classes: int = 3
    dropout: float = 0.3
    use_layer_norm: bool = True
    n_candle_steps: int = 5
    n_quantiles: int = 5  # q10, q25, q50, q75, q90
    
    # Progressive head enablement - start with just classification
    enable_quantile_head: bool = False
    enable_vol_state_head: bool = False
    enable_mu_head: bool = False
    enable_sigma_head: bool = False  # Most unstable - enable last
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class MultiHeadSimpleMLP(nn.Module):
    """
    Multi-head MLP with progressive complexity re-addition.
    
    Based on stable SimpleMLP, adds auxiliary heads one at a time.
    All outputs are clamped to prevent gradient explosions.
    
    Stability features inherited from SimpleMLP:
    1. LayerNorm after each layer
    2. Orthogonal initialization
    3. GELU activation
    4. Moderate dropout
    
    Additional stability for multi-head:
    5. Output clamping on all heads
    6. Small head networks (single linear layer)
    7. No shared layers between heads (isolated gradient paths)
    """
    
    def __init__(self, config: MultiHeadSimpleMLP_Config):
        super().__init__()
        self.config = config
        self.name = "MultiHeadSimpleMLP"
        
        # Store attributes for trainer compatibility
        self.input_dim = config.input_dim
        self.output_dim = config.num_classes
        self.hidden_dims = config.hidden_dims
        
        # Training metadata
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # Build shared trunk
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        self.trunk_dim = prev_dim
        
        # === HEAD 1: Classification (always enabled) ===
        self.classifier = nn.Linear(self.trunk_dim, config.num_classes)
        
        # === HEAD 2: Quantile (optional) ===
        if config.enable_quantile_head:
            self.quantile_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, config.n_quantiles)
            )
        else:
            self.quantile_head = None
            
        # === HEAD 3: Volatility State (optional) ===
        if config.enable_vol_state_head:
            self.vol_state_head = nn.Linear(self.trunk_dim, 3)  # CONTRACTION/NEUTRAL/EXPANSION
        else:
            self.vol_state_head = None
            
        # === HEAD 4: Mu/Expected Return (optional) ===
        if config.enable_mu_head:
            self.mu_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
        else:
            self.mu_head = None
            
        # === HEAD 5: Sigma/Uncertainty (optional, most unstable) ===
        if config.enable_sigma_head:
            self.sigma_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
        else:
            self.sigma_head = None
        
        self.n_candle_steps = config.n_candle_steps
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)  # Lower gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Multi-head forward pass with output clamping."""
        batch_size = x.size(0)
        device = x.device
        
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep
        
        # Shared trunk
        features = self.trunk(x)
        
        # === Classification (always active) ===
        class_logits = self.classifier(features)
        # Clamp logits to prevent extreme softmax
        class_logits = torch.clamp(class_logits, -10, 10)
        
        # === Quantile head ===
        if self.quantile_head is not None:
            quantiles = self.quantile_head(features)
            # Clamp to reasonable return range (±10%)
            quantiles = torch.clamp(quantiles, -0.1, 0.1)
            # Enforce monotonicity: q10 <= q25 <= q50 <= q75 <= q90
            quantiles = torch.cumsum(F.softplus(quantiles) * 0.01, dim=-1) - 0.025
        else:
            quantiles = torch.zeros(batch_size, 5, device=device)
        
        # === Vol state head ===
        if self.vol_state_head is not None:
            vol_state_logits = self.vol_state_head(features)
            vol_state_logits = torch.clamp(vol_state_logits, -10, 10)
        else:
            vol_state_logits = None
        
        # === Mu head ===
        if self.mu_head is not None:
            mu = self.mu_head(features)
            # Clamp expected return to ±5%
            mu = torch.clamp(mu, -0.05, 0.05)
        else:
            mu = torch.zeros(batch_size, 1, device=device)
        
        # === Sigma head ===
        if self.sigma_head is not None:
            log_sigma = self.sigma_head(features)
            # Clamp log_sigma to prevent extreme values
            log_sigma = torch.clamp(log_sigma, -8, 2)
            sigma = torch.exp(log_sigma)
        else:
            sigma = None
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=None,
            sl_distance=None,
            tp_distance=None,
            candle_deltas=None,
            vol_state_logits=vol_state_logits,
            acceleration=None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward returning just class logits."""
        return self.forward_multihead(x).class_logits
    
    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters(self) -> int:
        """Alias for compatibility."""
        return self.parameters_count()
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'config': {
                'enable_quantile_head': self.config.enable_quantile_head,
                'enable_vol_state_head': self.config.enable_vol_state_head,
                'enable_mu_head': self.config.enable_mu_head,
                'enable_sigma_head': self.config.enable_sigma_head,
            },
            'created_at': self.created_at,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epochs_trained
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: str = 'cuda'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_trained = checkpoint.get('epochs_trained', 0)


def create_multihead_simple_mlp(
    input_dim: int = 41,
    enable_quantile: bool = True,
    enable_vol_state: bool = False,
    enable_mu: bool = False,
    enable_sigma: bool = False
) -> MultiHeadSimpleMLP:
    """
    Factory function for MultiHeadSimpleMLP with configurable heads.
    
    Recommended enablement order (from most to least stable):
    1. Classification only (default SimpleMLP)
    2. + Quantile head
    3. + Vol state head  
    4. + Mu head
    5. + Sigma head (most unstable, enable last)
    """
    config = MultiHeadSimpleMLP_Config(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        enable_quantile_head=enable_quantile,
        enable_vol_state_head=enable_vol_state,
        enable_mu_head=enable_mu,
        enable_sigma_head=enable_sigma
    )
    return MultiHeadSimpleMLP(config)


if __name__ == "__main__":
    # Quick test
    model = create_simple_mlp(input_dim=41)
    print(f"SimpleMLP parameters: {model.parameters_count():,}")
    
    # Test forward pass
    x = torch.randn(32, 100, 41)  # batch=32, seq=100, features=41
    output = model.forward_multihead(x)
    print(f"class_logits shape: {output.class_logits.shape}")
    print(f"mu shape: {output.mu.shape}")
    
    # Test gradient flow
    loss = output.class_logits.sum()
    loss.backward()
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5
    print(f"Gradient norm: {grad_norm:.4f}")


# =============================================================================
# Enhanced MultiHeadMLP: Deeper architecture with residual connections
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow in deeper networks."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3, use_layer_norm: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Main path
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (project if dimensions differ)
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Add skip connection
        skip = self.skip(x)
        return out + skip


@dataclass  
class EnhancedMultiHeadMLP_Config:
    """Configuration for EnhancedMultiHeadMLP with deeper architecture."""
    input_dim: int = 41
    hidden_dims: list = None
    num_classes: int = 3
    dropout: float = 0.3
    use_layer_norm: bool = True
    use_residual: bool = True  # Enable residual connections
    n_candle_steps: int = 5
    n_quantiles: int = 5
    
    # Progressive head enablement
    enable_quantile_head: bool = True
    enable_vol_state_head: bool = True
    enable_mu_head: bool = True
    enable_sigma_head: bool = True
    enable_enter_head: bool = False  # Binary entry quality head
    enable_value_head: bool = False  # E[net R] regression head
    enable_edge_head: bool = False   # Edge regression head (net MFE - MAE proxy)
    enable_dir_head: bool = False    # v4.6 direction head (binary LONG/SHORT)
    enable_htf_head: bool = False    # v4.6 HTF score head (4-class)

    # v4.9 Distributional heads
    enable_win_head: bool = False    # p(R>0) binary head
    enable_dist_quantile_head: bool = False  # q10/q50/q90 quantile head (3 outputs)
    enable_regime_head: bool = False  # chop/trend/highvol classification (3-class)

    n_symbols: int = 1  # Number of distinct symbols for multi-asset embedding
    symbol_embed_dim: int = 8  # Embedding dimension per symbol
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Deeper architecture for more capacity
            self.hidden_dims = [512, 256, 128, 64]


class EnhancedMultiHeadMLP(nn.Module):
    """
    Enhanced Multi-head MLP with deeper architecture and residual connections.
    
    Improvements over MultiHeadSimpleMLP:
    1. Deeper network: [512, 256, 128, 64] vs [256, 128, 64]
    2. Residual/skip connections for better gradient flow
    3. Larger head networks for more expressive power
    
    All stability features preserved:
    - LayerNorm after each layer
    - Orthogonal initialization (low gain)
    - GELU activation
    - Output clamping on all heads
    - Moderate dropout
    """
    
    def __init__(self, config: EnhancedMultiHeadMLP_Config):
        super().__init__()
        self.config = config
        self.name = "EnhancedMultiHeadMLP"
        
        # Store attributes for trainer compatibility
        self.input_dim = config.input_dim
        self.output_dim = config.num_classes
        self.hidden_dims = config.hidden_dims
        
        # Training metadata
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # Symbol embedding for multi-asset training
        if config.n_symbols > 1:
            self.symbol_embedding = nn.Embedding(config.n_symbols, config.symbol_embed_dim)
            trunk_input_dim = config.input_dim + config.symbol_embed_dim
        else:
            self.symbol_embedding = None
            trunk_input_dim = config.input_dim
        
        # Build trunk with residual connections
        trunk_layers = []
        prev_dim = trunk_input_dim
        
        for hidden_dim in config.hidden_dims:
            if config.use_residual:
                trunk_layers.append(ResidualBlock(
                    prev_dim, hidden_dim, 
                    dropout=config.dropout, 
                    use_layer_norm=config.use_layer_norm
                ))
            else:
                trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
                if config.use_layer_norm:
                    trunk_layers.append(nn.LayerNorm(hidden_dim))
                trunk_layers.append(nn.GELU())
                trunk_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_dim = prev_dim
        
        # === HEAD 1: Classification (always enabled) ===
        self.classifier = nn.Sequential(
            nn.Linear(self.trunk_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, config.num_classes)
        )
        
        # === HEAD 2: Quantile ===
        if config.enable_quantile_head:
            self.quantile_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, config.n_quantiles)
            )
        else:
            self.quantile_head = None
            
        # === HEAD 3: Volatility State ===
        if config.enable_vol_state_head:
            self.vol_state_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 3)
            )
        else:
            self.vol_state_head = None
            
        # === HEAD 4: Mu/Expected Return ===
        if config.enable_mu_head:
            self.mu_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.mu_head = None
            
        # === HEAD 5: Sigma/Uncertainty ===
        if config.enable_sigma_head:
            self.sigma_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.sigma_head = None
        
        # === HEAD 6: Entry Quality (binary) ===
        if config.enable_enter_head:
            self.enter_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.enter_head = None
        
        # === HEAD 7: Value Head (E[net R] regression) ===
        if config.enable_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.value_head = None
        
        # === HEAD 8: Edge Head (net MFE - MAE quality regression) ===
        if config.enable_edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.edge_head = None
        
        # === HEAD 9: Direction Head (v4.6 binary LONG/SHORT) ===
        if config.enable_dir_head:
            self.dir_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.dir_head = None
        
        # === HEAD 10: HTF Score Head (v4.6 4-class classification) ===
        if config.enable_htf_head:
            self.htf_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 4)
            )
        else:
            self.htf_head = None

        # === HEAD 11: Win Head (v4.9 p(R>0) binary) ===
        if getattr(config, 'enable_win_head', False):
            self.win_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
        else:
            self.win_head = None

        # === HEAD 12: Distributional Quantile Head (v4.9 q10/q50/q90) ===
        if getattr(config, 'enable_dist_quantile_head', False):
            self.dist_quantile_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)
            )
        else:
            self.dist_quantile_head = None

        # === HEAD 13: Regime Head (v4.9 chop/trend/highvol 3-class) ===
        if getattr(config, 'enable_regime_head', False):
            self.regime_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 3)
            )
        else:
            self.regime_head = None
        
        self.n_candle_steps = config.n_candle_steps
        self._init_weights()
        
        # Store head_config for loss wiring
        self.head_config = {
            'enable_quantile': config.enable_quantile_head,
            'enable_vol_state': config.enable_vol_state_head,
            'enable_mu': config.enable_mu_head,
            'enable_sigma': config.enable_sigma_head,
            'enable_enter': config.enable_enter_head,
            'enable_value': config.enable_value_head,
            'enable_edge': config.enable_edge_head,
            'enable_dir': config.enable_dir_head,
            'enable_htf': config.enable_htf_head,
            'enable_win': getattr(config, 'enable_win_head', False),
            'enable_dist_quantile': getattr(config, 'enable_dist_quantile_head', False),
            'enable_regime': getattr(config, 'enable_regime_head', False),
        }
    
    def _init_weights(self):
        """Orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simple forward for classification only (legacy trainer compatibility)."""
        if x.dim() == 3:
            x = x[:, -1, :]
        if self.symbol_embedding is not None and symbol_ids is not None:
            sym_emb = self.symbol_embedding(symbol_ids)
            x = torch.cat([x, sym_emb], dim=-1)
        features = self.trunk(x)
        logits = self.classifier(features)
        return torch.clamp(logits, -10, 10)
    
    def forward_multihead(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None) -> MultiHeadOutput:
        """Multi-head forward pass with output clamping."""
        batch_size = x.size(0)
        device = x.device
        
        if x.dim() == 3:
            x = x[:, -1, :]
        
        if self.symbol_embedding is not None and symbol_ids is not None:
            sym_emb = self.symbol_embedding(symbol_ids)
            x = torch.cat([x, sym_emb], dim=-1)
        
        # Shared trunk
        features = self.trunk(x)
        
        # === Classification ===
        class_logits = self.classifier(features)
        class_logits = torch.clamp(class_logits, -10, 10)
        
        # === Quantile ===
        if self.quantile_head is not None:
            quantiles_raw = self.quantile_head(features)
            quantiles_raw = torch.clamp(quantiles_raw, -0.5, 0.5)
            # Enforce monotonicity with bounded output
            quantiles = torch.cumsum(F.softplus(quantiles_raw * 0.1) + 1e-4, dim=-1)  # Scale down input
            quantiles = quantiles - quantiles.mean(dim=-1, keepdim=True)
            quantiles = torch.clamp(quantiles, -0.1, 0.1)  # Clamp to target scale
        else:
            quantiles = torch.zeros(batch_size, self.config.n_quantiles, device=device)
        
        # === Mu ===
        if self.mu_head is not None:
            mu = self.mu_head(features)
            mu = torch.clamp(mu, -0.1, 0.1)
        else:
            mu = torch.zeros(batch_size, 1, device=device)
        
        # === Sigma ===
        if self.sigma_head is not None:
            log_sigma = self.sigma_head(features)
            log_sigma = torch.clamp(log_sigma, -5, 2)
            sigma = F.softplus(log_sigma) + 1e-6
        else:
            sigma = torch.ones(batch_size, 1, device=device) * 0.01
        
        # === Vol State ===
        if self.vol_state_head is not None:
            vol_state_logits = self.vol_state_head(features)
            vol_state_logits = torch.clamp(vol_state_logits, -10, 10)
        else:
            vol_state_logits = torch.zeros(batch_size, 3, device=device)
        
        # === Enter Quality ===
        if self.enter_head is not None:
            enter_logits = self.enter_head(features)
            enter_logits = torch.clamp(enter_logits, -5, 5)
        else:
            enter_logits = None
        
        # === Value Head (E[net R]) ===
        if self.value_head is not None:
            value_logits = self.value_head(features)
            value_logits = torch.clamp(value_logits, -3.0, 3.0)
        else:
            value_logits = None
        
        # === Edge Head (net MFE - MAE quality) ===
        edge_head = getattr(self, 'edge_head', None)
        if edge_head is not None:
            edge_logits = edge_head(features)
            edge_logits = torch.clamp(edge_logits, -5.0, 5.0)
        else:
            edge_logits = None
        
        # === Direction Head (v4.6 LONG/SHORT) ===
        dir_head = getattr(self, 'dir_head', None)
        if dir_head is not None:
            dir_logits = dir_head(features)
            dir_logits = torch.clamp(dir_logits, -5.0, 5.0)
        else:
            dir_logits = None
        
        # === HTF Score Head (v4.6 4-class) ===
        htf_head = getattr(self, 'htf_head', None)
        if htf_head is not None:
            htf_logits = htf_head(features)
            htf_logits = torch.clamp(htf_logits, -10.0, 10.0)
        else:
            htf_logits = None
        
        # === Win Head (v4.9 p(R>0)) ===
        win_head = getattr(self, 'win_head', None)
        if win_head is not None:
            win_logits = win_head(features)
            win_logits = torch.clamp(win_logits, -5.0, 5.0)
        else:
            win_logits = None

        # === Distributional Quantile Head (v4.9 q10/q50/q90) ===
        dist_quantile_head = getattr(self, 'dist_quantile_head', None)
        if dist_quantile_head is not None:
            dq_raw = dist_quantile_head(features)
            q10_raw = dq_raw[:, 0:1]
            delta_50 = torch.nn.functional.softplus(dq_raw[:, 1:2])
            delta_90 = torch.nn.functional.softplus(dq_raw[:, 2:3])
            q10 = torch.clamp(q10_raw, -5.0, 5.0)
            q50 = torch.clamp(q10 + delta_50, -5.0, 5.0)
            q90 = torch.clamp(q50 + delta_90, -5.0, 5.0)
            dist_q_raw = torch.cat([q10, q50, q90], dim=-1)
        else:
            dist_q_raw = None

        # === Regime Head (v4.9 chop/trend/highvol) ===
        regime_head_mod = getattr(self, 'regime_head', None)
        if regime_head_mod is not None:
            regime_logits = regime_head_mod(features)
            regime_logits = torch.clamp(regime_logits, -10.0, 10.0)
        else:
            regime_logits = None

        # Placeholders for unused heads
        entry_offset = torch.zeros(batch_size, 1, device=device)
        sl_distance = torch.ones(batch_size, 1, device=device) * 0.01
        tp_distance = torch.ones(batch_size, 1, device=device) * 0.02
        candle_deltas = torch.zeros(batch_size, self.n_candle_steps, 3, device=device)
        acceleration = torch.zeros(batch_size, 1, device=device)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            sigma=sigma,
            quantiles=quantiles,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas,
            vol_state_logits=vol_state_logits,
            acceleration=acceleration,
            enter_logits=enter_logits,
            value_logits=value_logits,
            edge_logits=edge_logits,
            dir_logits=dir_logits,
            htf_logits=htf_logits,
            win_logits=win_logits,
            dist_quantiles=dist_q_raw,
            regime_logits=regime_logits,
        )
    
    def parameters_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model and config."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'name': self.name,
            'created_at': self.created_at,
            'training_history': self.training_history,
            'epochs_trained': self.epochs_trained,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """Load model from checkpoint with backward compatibility.
        
        v4.5 checkpoints missing dir_head/htf_head will load successfully
        with those heads disabled (None).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        if not hasattr(config, 'enable_dir_head'):
            config.enable_dir_head = False
        if not hasattr(config, 'enable_htf_head'):
            config.enable_htf_head = False
        if not hasattr(config, 'enable_win_head'):
            config.enable_win_head = False
        if not hasattr(config, 'enable_dist_quantile_head'):
            config.enable_dist_quantile_head = False
        if not hasattr(config, 'enable_regime_head'):
            config.enable_regime_head = False
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.created_at = checkpoint.get('created_at', '')
        model.training_history = checkpoint.get('training_history', [])
        model.epochs_trained = checkpoint.get('epochs_trained', 0)
        model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return model
