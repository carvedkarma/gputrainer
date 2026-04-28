import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base import BaseModel

class MetaLearner(BaseModel):
    def __init__(
        self,
        num_models: int,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("meta_learner", num_models * output_dim + feature_dim, output_dim)
        
        self.num_models = num_models
        self.feature_dim = feature_dim
        
        self.model_encoder = nn.Sequential(
            nn.Linear(num_models * output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1)
        )
        
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, model_outputs: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model_encoded = self.model_encoder(model_outputs.view(model_outputs.size(0), -1))
        feature_encoded = self.feature_encoder(features)
        
        combined = torch.cat([model_encoded, feature_encoded], dim=-1)
        
        weights = self.weight_predictor(combined)
        
        model_outputs_reshaped = model_outputs.view(model_outputs.size(0), self.num_models, -1)
        weighted_output = torch.einsum('bn,bno->bo', weights, model_outputs_reshaped)
        
        residual = self.residual(combined)
        
        final_output = weighted_output + 0.1 * residual
        
        return final_output, weights


class DeepEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], device: str = "cuda"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(F.softmax(output, dim=-1))
                
        stacked = torch.stack(outputs, dim=0)
        
        mean_pred = stacked.mean(dim=0)
        uncertainty = stacked.std(dim=0).mean(dim=-1)
        
        return mean_pred, uncertainty
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_pred, uncertainty = self.forward(x)
        
        predicted_class = torch.argmax(mean_pred, dim=-1)
        confidence = mean_pred.max(dim=-1).values
        
        return predicted_class, confidence, uncertainty


class AttentionEnsemble(BaseModel):
    def __init__(
        self,
        num_models: int,
        model_output_dim: int = 3,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__("attention_ensemble", num_models * model_output_dim + feature_dim, model_output_dim)
        
        self.num_models = num_models
        self.model_output_dim = model_output_dim
        
        self.model_projection = nn.Linear(model_output_dim, hidden_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, model_output_dim)
        )
        
    def forward(self, model_outputs: List[torch.Tensor], features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        model_embeddings = []
        for output in model_outputs:
            proj = self.model_projection(output)
            model_embeddings.append(proj)
            
        model_seq = torch.stack(model_embeddings, dim=1)
        
        feature_proj = self.feature_projection(features).unsqueeze(1)
        
        cross_attn_out, _ = self.cross_attention(feature_proj, model_seq, model_seq)
        
        combined = torch.cat([model_seq, cross_attn_out], dim=1)
        self_attn_out, attn_weights = self.self_attention(combined, combined, combined)
        
        pooled = self_attn_out.mean(dim=1)
        
        output = self.output(pooled)
        return output


class MasterEnsemble(BaseModel):
    def __init__(
        self,
        model_configs: Dict[str, Dict],
        feature_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 3
    ):
        super().__init__("master_ensemble", feature_dim, output_dim)
        
        self.model_names = list(model_configs.keys())
        self.num_models = len(self.model_names)
        
        self.meta_learner = MetaLearner(
            self.num_models, feature_dim, hidden_dim, dropout, output_dim
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.num_models * output_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.num_models * output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.model_performance = {name: {"accuracy": 0.5, "ema_accuracy": 0.5} 
                                  for name in self.model_names}
        self.ema_alpha = 0.1
        
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
                features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs_list = [model_outputs[name] for name in self.model_names]
        stacked_outputs = torch.stack(outputs_list, dim=1)
        
        ensemble_output, model_weights = self.meta_learner(stacked_outputs, features)
        
        flat_outputs = stacked_outputs.view(stacked_outputs.size(0), -1)
        combined = torch.cat([flat_outputs, features], dim=-1)
        
        confidence = self.confidence_estimator(combined)
        uncertainty = self.uncertainty_estimator(flat_outputs)
        
        return {
            "prediction": ensemble_output,
            "model_weights": model_weights,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "individual_outputs": model_outputs
        }
    
    def update_model_performance(self, model_name: str, correct: bool):
        if model_name in self.model_performance:
            current = self.model_performance[model_name]["ema_accuracy"]
            new_value = 1.0 if correct else 0.0
            self.model_performance[model_name]["ema_accuracy"] = (
                self.ema_alpha * new_value + (1 - self.ema_alpha) * current
            )
    
    def get_action(self, model_outputs: Dict[str, torch.Tensor],
                   features: torch.Tensor) -> Tuple[int, float, Dict]:
        result = self.forward(model_outputs, features)
        
        probs = F.softmax(result["prediction"], dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = result["confidence"].item()
        
        ev_long = probs[0, 0].item()
        ev_short = probs[0, 1].item()
        ev_hold = probs[0, 2].item()
        
        if confidence < 0.3 or result["uncertainty"].item() > 0.5:
            action = 2
            
        info = {
            "probabilities": probs[0].tolist(),
            "model_weights": result["model_weights"][0].tolist(),
            "uncertainty": result["uncertainty"].item(),
            "ev_long": ev_long,
            "ev_short": ev_short,
            "ev_hold": ev_hold
        }
        
        return action, confidence, info


class OnlineLearningEnsemble(MasterEnsemble):
    def __init__(
        self,
        model_configs: Dict[str, Dict],
        feature_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 3,
        learning_rate: float = 1e-4
    ):
        super().__init__(model_configs, feature_dim, hidden_dim, dropout, output_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.recent_predictions = []
        self.max_history = 1000
        
    def online_update(self, features: torch.Tensor, model_outputs: Dict[str, torch.Tensor],
                      actual_outcome: int) -> float:
        self.train()
        
        result = self.forward(model_outputs, features)
        
        target = torch.LongTensor([actual_outcome]).to(features.device)
        loss = F.cross_entropy(result["prediction"], target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        predicted = torch.argmax(result["prediction"], dim=-1).item()
        for i, name in enumerate(self.model_names):
            model_pred = torch.argmax(model_outputs[name], dim=-1).item()
            self.update_model_performance(name, model_pred == actual_outcome)
            
        self.recent_predictions.append({
            "predicted": predicted,
            "actual": actual_outcome,
            "correct": predicted == actual_outcome
        })
        
        if len(self.recent_predictions) > self.max_history:
            self.recent_predictions.pop(0)
            
        self.eval()
        return loss.item()
    
    def get_recent_accuracy(self, window: int = 100) -> float:
        recent = self.recent_predictions[-window:]
        if not recent:
            return 0.5
        return sum(p["correct"] for p in recent) / len(recent)
