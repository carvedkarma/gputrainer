import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from .base import BaseModel

class SentimentEncoder(BaseModel):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 3,
        freeze_bert: bool = False
    ):
        super().__init__("sentiment_encoder", 768, output_dim)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        bert_dim = self.bert.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
    def encode_text(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.get_embeddings(input_ids, attention_mask)
        projected = self.projection(embeddings)
        logits = self.classifier(projected)
        return logits
    
    def predict_from_text(self, texts: List[str], device: str = "cuda") -> torch.Tensor:
        self.eval()
        encoded = self.encode_text(texts)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            
        return probs


class MultiModalSentiment(BaseModel):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        price_input_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("multimodal_sentiment", price_input_dim + 768, output_dim)
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        text_dim = self.text_encoder.config.hidden_size
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(price_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                price_features: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_projected = self.text_projection(text_features)
        
        price_projected = self.price_encoder(price_features)
        
        text_seq = text_projected.unsqueeze(1)
        price_seq = price_projected.unsqueeze(1)
        
        attended, _ = self.cross_attention(text_seq, price_seq, price_seq)
        attended = attended.squeeze(1)
        
        combined = torch.cat([attended, price_projected], dim=-1)
        fused = self.fusion(combined)
        
        logits = self.classifier(fused)
        return logits


class NewsAggregator(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        max_headlines: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.max_headlines = max_headlines
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        text_dim = self.text_encoder.config.hidden_size
        
        self.headline_projection = nn.Linear(text_dim, hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, headlines_batch: List[List[str]], device: str = "cuda") -> torch.Tensor:
        batch_embeddings = []
        
        for headlines in headlines_batch:
            if not headlines:
                headlines = ["No news available"]
                
            headlines = headlines[:self.max_headlines]
            
            encoded = self.tokenizer(
                headlines,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
            projected = self.headline_projection(embeddings)
            
            attn_weights = self.attention(projected)
            attn_weights = F.softmax(attn_weights, dim=0)
            
            aggregated = torch.sum(attn_weights * projected, dim=0)
            batch_embeddings.append(aggregated)
            
        batch_tensor = torch.stack(batch_embeddings)
        output = self.output_projection(batch_tensor)
        
        return output


class SentimentPricePredictor(BaseModel):
    def __init__(
        self,
        price_input_dim: int = 64,
        hidden_dim: int = 256,
        num_sentiment_features: int = 5,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("sentiment_price_predictor", price_input_dim + num_sentiment_features, output_dim)
        
        self.price_encoder = nn.Sequential(
            nn.Linear(price_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(num_sentiment_features, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        combined_dim = hidden_dim // 2 + hidden_dim // 4
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_dim // 4, output_dim)
        
        self.sentiment_gate = nn.Sequential(
            nn.Linear(num_sentiment_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, price_features: torch.Tensor, 
                sentiment_features: torch.Tensor) -> torch.Tensor:
        price_encoded = self.price_encoder(price_features)
        sentiment_encoded = self.sentiment_encoder(sentiment_features)
        
        sentiment_weight = self.sentiment_gate(sentiment_features)
        sentiment_encoded = sentiment_encoded * sentiment_weight
        
        combined = torch.cat([price_encoded, sentiment_encoded], dim=-1)
        fused = self.fusion(combined)
        
        logits = self.classifier(fused)
        return logits
