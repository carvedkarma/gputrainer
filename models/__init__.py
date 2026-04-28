"""
Neural Network Models for BTC Futures Trading

This module contains all the deep learning architectures:
- Transformer: Attention-based sequence learning
- LSTM: Recurrent networks for time series
- CNN: Convolutional networks for pattern recognition
- VAE: Variational autoencoders for representation learning
- GNN: Graph neural networks for cross-asset modeling
- RL: Reinforcement learning agents
- Sentiment: NLP models for news/social media
- Ensemble: Meta-learning and model combination
"""

from .base import BaseModel, PositionalEncoding, AttentionBlock, ResidualBlock
from .transformer import TransformerPriceModel, TemporalFusionTransformer
from .lstm import BidirectionalLSTM, StackedLSTM, ConvLSTM
from .cnn import ResNetPrice, InceptionNet, WaveNet
from .vae import MarketVAE, BetaVAE, ConditionalVAE
from .gnn import CrossAssetGNN, TemporalGNN
from .rl_agent import PPOAgent, TradingEnvironment, ActorCritic
try:
    from .sentiment import SentimentEncoder, MultiModalSentiment, SentimentPricePredictor
    _HAVE_SENTIMENT = True
except ImportError:
    _HAVE_SENTIMENT = False
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "[MODELS] sentiment sub-module unavailable (likely missing 'transformers' or "
        "'torch' dependency).  SentimentEncoder / MultiModalSentiment / "
        "SentimentPricePredictor will not be exported.  "
        "Install with: pip install transformers"
    )
from .ensemble import MetaLearner, DeepEnsemble, MasterEnsemble, OnlineLearningEnsemble
from .multihead import (
    MultiHeadOutput, MultiHeadTransformer, MultiScaleTransformer, MultiHeadTFT, MultiHeadLSTM, 
    MultiHeadCNN, MultiHeadGNN, MultiHeadVAE, get_multihead_model,
    MultiScaleTemporalEmbedding, MultiScaleAttentionBlock
)

__all__ = [
    # Base
    "BaseModel",
    "PositionalEncoding", 
    "AttentionBlock",
    "ResidualBlock",
    
    # Transformer
    "TransformerPriceModel",
    "TemporalFusionTransformer",
    
    # LSTM
    "BidirectionalLSTM",
    "StackedLSTM",
    "ConvLSTM",
    
    # CNN
    "ResNetPrice",
    "InceptionNet",
    "WaveNet",
    
    # VAE
    "MarketVAE",
    "BetaVAE",
    "ConditionalVAE",
    
    # GNN
    "CrossAssetGNN",
    "TemporalGNN",
    
    # RL
    "PPOAgent",
    "TradingEnvironment",
    "ActorCritic",
    
    # Sentiment (conditional — only present when transformers is installed)
    *( ["SentimentEncoder", "MultiModalSentiment", "SentimentPricePredictor"]
       if _HAVE_SENTIMENT else [] ),
    
    # Ensemble
    "MetaLearner",
    "DeepEnsemble",
    "MasterEnsemble",
    "OnlineLearningEnsemble",
    
    # Multi-Head Models
    "MultiHeadOutput",
    "MultiHeadTransformer",
    "MultiScaleTransformer",
    "MultiScaleTemporalEmbedding",
    "MultiScaleAttentionBlock",
    "MultiHeadTFT",
    "MultiHeadLSTM",
    "MultiHeadCNN",
    "MultiHeadGNN",
    "MultiHeadVAE",
    "get_multihead_model",
]
