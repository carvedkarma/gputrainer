import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .base import BaseModel

class MarketVAE(BaseModel):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 100,
        latent_dim: int = 64,
        hidden_dims: list = [128, 256, 512],
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("market_vae", input_dim, output_dim)
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        encoder_layers = []
        in_features = input_dim * sequence_length
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim * sequence_length))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, output_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, self.sequence_length, self.input_dim)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        logits = self.classifier(z)
        return logits
    
    def forward_full(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        logits = self.classifier(z)
        return x_recon, mu, log_var, logits
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
    
    @staticmethod
    def vae_loss(x: torch.Tensor, x_recon: torch.Tensor, 
                 mu: torch.Tensor, log_var: torch.Tensor,
                 beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class BetaVAE(MarketVAE):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 100,
        latent_dim: int = 64,
        hidden_dims: list = [128, 256, 512],
        dropout: float = 0.2,
        output_dim: int = 3,
        beta: float = 4.0
    ):
        super().__init__(input_dim, sequence_length, latent_dim, hidden_dims, dropout, output_dim)
        self.name = "beta_vae"
        self.beta = beta


class ConditionalVAE(BaseModel):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        sequence_length: int = 100,
        latent_dim: int = 64,
        hidden_dims: list = [128, 256, 512],
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("conditional_vae", input_dim, output_dim)
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        encoder_layers = []
        in_features = input_dim * sequence_length + 64
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        in_features = latent_dim + 64
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim * sequence_length))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 64, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim)
        )
        
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        c_encoded = self.condition_encoder(c)
        
        h = torch.cat([x_flat, c_encoded], dim=1)
        h = self.encoder(h)
        
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_encoded = self.condition_encoder(c)
        z_c = torch.cat([z, c_encoded], dim=1)
        x_recon = self.decoder(z_c)
        x_recon = x_recon.view(-1, self.sequence_length, self.input_dim)
        return x_recon
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        c_encoded = self.condition_encoder(c)
        z_c = torch.cat([z, c_encoded], dim=1)
        logits = self.classifier(z_c)
        return logits
