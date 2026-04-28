"""
Self-Supervised Pretraining for Trading Models

Foundation model pretraining on billions of timesteps:
1. Masked Time-Series Modeling: Mask segments, predict them
2. Next-K-Step Distribution Prediction: Predict future distribution
3. Contrastive Learning: Same regime, different assets
4. Regime Clustering: Learn regime representations

"This is where GPUs matter - pretrain an encoder, then finetune on trading objective."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PretrainingConfig:
    """Configuration for pretraining."""
    mask_ratio: float = 0.15
    mask_length: int = 10
    prediction_horizon: int = 12
    contrastive_temp: float = 0.07
    encoder_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1


class MaskedTimeSeriesEncoder(nn.Module):
    """
    Encoder that learns representations from masked time-series prediction.
    
    Similar to BERT but for time-series:
    - Mask random segments of the input sequence
    - Predict the masked values
    - Learn useful representations for downstream tasks
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
    def create_mask(self, 
                    batch_size: int, 
                    seq_len: int, 
                    mask_ratio: float = 0.15,
                    mask_length: int = 10,
                    device: str = "cuda") -> torch.Tensor:
        """
        Create span masking pattern.
        
        Instead of random token masking, we mask contiguous spans
        which is more appropriate for time-series data.
        """
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        n_masks = max(1, int(seq_len * mask_ratio / mask_length))
        
        for b in range(batch_size):
            for _ in range(n_masks):
                start = np.random.randint(0, max(1, seq_len - mask_length))
                end = min(start + mask_length, seq_len)
                mask[b, start:end] = True
        
        return mask
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional masking.
        
        Args:
            x: Input sequence [batch, seq, features]
            mask: Boolean mask [batch, seq], True = masked
            return_hidden: Whether to return hidden states
            
        Returns:
            reconstruction: Reconstructed input [batch, seq, features]
            hidden: Hidden representations [batch, seq, hidden_dim]
            mask: The mask that was applied
        """
        batch_size, seq_len, _ = x.shape
        
        if mask is None:
            mask = self.create_mask(batch_size, seq_len, device=x.device)
        
        hidden = self.input_projection(x)
        
        hidden = hidden + self.pos_embedding[:, :seq_len, :]
        
        mask_expanded = mask.unsqueeze(-1).expand_as(hidden)
        hidden = torch.where(mask_expanded, self.mask_token.expand_as(hidden), hidden)
        
        hidden = self.transformer(hidden)
        
        reconstruction = self.reconstruction_head(hidden)
        
        if return_hidden:
            return reconstruction, hidden, mask
        return reconstruction, hidden, mask
    
    def compute_loss(self, 
                     x: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        reconstruction, _, mask = self.forward(x, mask)
        
        mse = F.mse_loss(reconstruction[mask], x[mask], reduction='mean')
        
        return mse
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder representations without masking."""
        hidden = self.input_projection(x)
        hidden = hidden + self.pos_embedding[:, :x.shape[1], :]
        hidden = self.transformer(hidden)
        return hidden


class NextStepDistributionPredictor(nn.Module):
    """
    Predicts the distribution of next-k-steps, not just the mean.
    
    Outputs quantiles (p10, p25, p50, p75, p90) of the future return distribution.
    This teaches the model about uncertainty and tail risks.
    """
    
    def __init__(self,
                 encoder: MaskedTimeSeriesEncoder,
                 prediction_horizon: int = 12,
                 n_quantiles: int = 5):
        super().__init__()
        
        self.encoder = encoder
        self.prediction_horizon = prediction_horizon
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_quantiles = n_quantiles
        
        hidden_dim = encoder.hidden_dim
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict future return quantiles.
        
        Args:
            x: Input sequence [batch, seq, features]
            
        Returns:
            quantile_predictions: [batch, n_quantiles]
        """
        hidden = self.encoder.encode(x)
        
        last_hidden = hidden[:, -1, :]
        
        quantiles = self.prediction_head(last_hidden)
        
        quantiles = torch.sort(quantiles, dim=-1)[0]
        
        return quantiles
    
    def compute_loss(self, 
                     x: torch.Tensor,
                     future_returns: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile regression loss.
        
        Uses pinball loss for each quantile.
        """
        quantile_preds = self.forward(x)
        
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            errors = future_returns.squeeze() - quantile_preds[:, i]
            pinball = torch.where(
                errors >= 0,
                q * errors,
                (q - 1) * errors
            )
            total_loss += pinball.mean()
        
        return total_loss / len(self.quantiles)


class ContrastiveLearner(nn.Module):
    """
    Contrastive learning across assets in the same regime.
    
    Idea: Different assets in the same market regime should have 
    similar representations. This teaches the model to identify 
    market conditions independent of the specific asset.
    """
    
    def __init__(self,
                 encoder: MaskedTimeSeriesEncoder,
                 projection_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        hidden_dim = encoder.hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized projection for contrastive learning."""
        hidden = self.encoder.encode(x)
        pooled = hidden.mean(dim=1)  # Global average pooling
        projection = self.projector(pooled)
        return F.normalize(projection, dim=-1)
    
    def compute_loss(self,
                     x1: torch.Tensor,
                     x2: torch.Tensor,
                     same_regime: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            x1, x2: Two views of data [batch, seq, features]
            same_regime: Boolean tensor indicating if pairs are same regime [batch]
        """
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class RegimeClusteringModule(nn.Module):
    """
    Self-supervised regime clustering.
    
    Learns to cluster similar market conditions without labels.
    Uses deep clustering with auxiliary target updates.
    """
    
    def __init__(self,
                 encoder: MaskedTimeSeriesEncoder,
                 n_clusters: int = 6,
                 alpha: float = 1.0):
        super().__init__()
        
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        hidden_dim = encoder.hidden_dim
        self.cluster_layer = nn.Linear(hidden_dim, n_clusters, bias=False)
        
        with torch.no_grad():
            self.cluster_layer.weight.data = F.normalize(
                torch.randn(n_clusters, hidden_dim), dim=1
            )
    
    def get_cluster_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get soft cluster assignments using Student's t-distribution."""
        hidden = self.encoder.encode(x)
        pooled = hidden.mean(dim=1)
        
        cluster_centers = self.cluster_layer.weight
        
        distances = torch.cdist(pooled, cluster_centers, p=2)
        
        q = 1 / (1 + distances ** 2 / self.alpha)
        q = q ** ((self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    def compute_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary target distribution."""
        f = q.sum(dim=0)
        p = q ** 2 / f
        p = p / p.sum(dim=1, keepdim=True)
        return p.detach()
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence clustering loss."""
        q = self.get_cluster_probs(x)
        p = self.compute_target_distribution(q)
        
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        return loss


class PretrainingPipeline:
    """
    Complete pretraining pipeline combining all self-supervised tasks.
    """
    
    def __init__(self,
                 input_dim: int,
                 config: PretrainingConfig = None,
                 device: str = "cuda"):
        self.config = config or PretrainingConfig()
        self.device = device
        
        self.encoder = MaskedTimeSeriesEncoder(
            input_dim=input_dim,
            hidden_dim=self.config.encoder_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        ).to(device)
        
        self.next_step_predictor = NextStepDistributionPredictor(
            encoder=self.encoder,
            prediction_horizon=self.config.prediction_horizon
        ).to(device)
        
        self.contrastive = ContrastiveLearner(
            encoder=self.encoder,
            temperature=self.config.contrastive_temp
        ).to(device)
        
        self.clustering = RegimeClusteringModule(
            encoder=self.encoder,
            n_clusters=6
        ).to(device)
        
        self.loss_weights = {
            "masked": 1.0,
            "next_step": 0.5,
            "contrastive": 0.3,
            "clustering": 0.2
        }
        
    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step with all pretraining tasks.
        """
        x = batch["features"].to(self.device)
        future_returns = batch.get("future_returns")
        if future_returns is not None:
            future_returns = future_returns.to(self.device)
        
        optimizer.zero_grad()
        
        losses = {}
        
        mask = self.encoder.create_mask(
            x.shape[0], x.shape[1],
            self.config.mask_ratio,
            self.config.mask_length,
            self.device
        )
        losses["masked"] = self.encoder.compute_loss(x, mask)
        
        if future_returns is not None:
            losses["next_step"] = self.next_step_predictor.compute_loss(x, future_returns)
        
        losses["clustering"] = self.clustering.compute_loss(x)
        
        total_loss = sum(self.loss_weights.get(k, 0) * v for k, v in losses.items())
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def get_pretrained_encoder(self) -> MaskedTimeSeriesEncoder:
        """Return the pretrained encoder for fine-tuning."""
        return self.encoder
    
    def save_pretrained(self, path: str):
        """Save pretrained encoder."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "config": self.config
        }, path)
        logger.info(f"Saved pretrained encoder to {path}")
    
    def load_pretrained(self, path: str):
        """Load pretrained encoder."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        logger.info(f"Loaded pretrained encoder from {path}")


class PretrainedRegressionModel(nn.Module):
    """
    Fine-tuning wrapper for pretrained encoder.
    
    Uses the pretrained encoder and adds regression heads for (mu, sigma).
    """
    
    def __init__(self,
                 pretrained_encoder: MaskedTimeSeriesEncoder,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = pretrained_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_dim = pretrained_encoder.hidden_dim
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for regression.
        
        Returns:
            mu: Expected return [batch, 1]
            sigma: Uncertainty [batch, 1]
        """
        hidden = self.encoder.encode(x)
        pooled = hidden[:, -1, :]  # Use last timestep
        
        mu = self.mu_head(pooled)
        sigma = self.sigma_head(pooled) + 0.001
        
        return mu, sigma
