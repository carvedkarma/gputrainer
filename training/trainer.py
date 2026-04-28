import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        mixed_precision: bool = False,  # Disabled by default - FP16 can cause NaN with class weights
        gui_mode: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        feature_scaler = None,
        feature_columns: Optional[list] = None,
        training_mode: str = "stf",
        horizon_periods: int = 16,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        self.gui_mode = gui_mode  # Disable tqdm in GUI mode to prevent UI freeze
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.mixed_precision = mixed_precision
        self.use_focal_loss = use_focal_loss
        
        # Store training config for checkpoint saving
        self.feature_scaler = feature_scaler  # sklearn StandardScaler for features
        self.feature_columns = feature_columns
        self.training_mode = training_mode
        self.horizon_periods = horizon_periods
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate * 10,
            epochs=config.training.epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Fixed: Use device-specific GradScaler to avoid deprecation warning
        self.scaler = GradScaler('cuda') if mixed_precision else None
        
        # Use Focal Loss or CrossEntropy for classification
        if use_focal_loss:
            from training.multihead_loss import FocalLoss
            self.criterion = FocalLoss(gamma=focal_gamma, num_classes=3)
            if class_weights is not None:
                self.criterion.set_alpha(class_weights.to(device))
            logger.info(f"[FOCAL] Using FocalLoss: gamma={focal_gamma}, class_weights={'yes' if class_weights is not None else 'no'}")
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.writer = SummaryWriter(config.training.log_dir)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.epoch_callback = None  # Callback for GUI progress updates
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Track gradient norms for stability monitoring
        grad_norms_pre = []
        grad_norms_post = []
        
        # Use tqdm only in non-GUI mode (tqdm floods stdout and freezes GUI)
        if self.gui_mode:
            loader = self.train_loader
        else:
            loader = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        nan_batch_count = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Initialize per-batch gradient norm tracking
            grad_norm_pre = None
            grad_norm_post = None
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast('cuda'):
                    output = self.model(data)
                    loss = self.criterion(output, target.long())
                
                # NaN detection with diagnostic logging
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batch_count += 1
                    # Diagnostic: identify WHERE NaN originates
                    x_nan = torch.isnan(data).any().item()
                    x_inf = torch.isinf(data).any().item()
                    out_nan = torch.isnan(output).any().item()
                    out_inf = torch.isinf(output).any().item()
                    if nan_batch_count <= 3:  # Only log first 3
                        logger.warning(f"NaN batch {nan_batch_count}: x_nan={x_nan}, x_inf={x_inf}, "
                                     f"logits_nan={out_nan}, logits_inf={out_inf}, "
                                     f"x_range=[{data.min().item():.4f}, {data.max().item():.4f}]")
                    if nan_batch_count > 10:
                        logger.warning(f"Too many NaN batches ({nan_batch_count}), stopping epoch")
                        break
                    continue
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Track gradient norm BEFORE clipping (mixed precision)
                total_norm_pre = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_pre += p.grad.data.norm(2).item() ** 2
                grad_norm_pre = total_norm_pre ** 0.5
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                
                # Track gradient norm AFTER clipping
                total_norm_post = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_post += p.grad.data.norm(2).item() ** 2
                grad_norm_post = total_norm_post ** 0.5
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target.long())
                
                # NaN detection with diagnostic logging
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batch_count += 1
                    # Diagnostic: identify WHERE NaN originates
                    x_nan = torch.isnan(data).any().item()
                    x_inf = torch.isinf(data).any().item()
                    out_nan = torch.isnan(output).any().item()
                    out_inf = torch.isinf(output).any().item()
                    if nan_batch_count <= 3:  # Only log first 3
                        logger.warning(f"NaN batch {nan_batch_count}: x_nan={x_nan}, x_inf={x_inf}, "
                                     f"logits_nan={out_nan}, logits_inf={out_inf}, "
                                     f"x_range=[{data.min().item():.4f}, {data.max().item():.4f}]")
                    if nan_batch_count > 10:
                        logger.warning(f"Too many NaN batches ({nan_batch_count}), stopping epoch")
                        break
                    continue
                    
                loss.backward()
                
                # Track gradient norm BEFORE clipping
                total_norm_pre = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_pre += p.grad.data.norm(2).item() ** 2
                grad_norm_pre = total_norm_pre ** 0.5
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                
                # Track gradient norm AFTER clipping
                total_norm_post = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm_post += p.grad.data.norm(2).item() ** 2
                grad_norm_post = total_norm_post ** 0.5
                
                self.optimizer.step()
                
            self.scheduler.step()
            
            # Accumulate gradient norms for epoch average (only if computed this batch)
            if grad_norm_pre is not None:
                grad_norms_pre.append(grad_norm_pre)
                grad_norms_post.append(grad_norm_post)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.long()).sum().item()
            total += target.size(0)
            
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
            if grad_norm_pre is not None:
                self.writer.add_scalar("train/grad_norm_pre", grad_norm_pre, self.global_step)
                self.writer.add_scalar("train/grad_norm_post", grad_norm_post, self.global_step)
            self.global_step += 1
            
            # Only update tqdm in non-GUI mode
            if not self.gui_mode:
                loader.set_postfix({
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                    "grad": f"{grad_norm_pre:.2f}" if grad_norm_pre is not None else "N/A"
                })
            
            # Yield to UI thread every batch to prevent GUI freeze
            time.sleep(0)
        
        # Compute epoch-level gradient norm statistics
        if grad_norms_pre:
            avg_grad_pre = sum(grad_norms_pre) / len(grad_norms_pre)
            avg_grad_post = sum(grad_norms_post) / len(grad_norms_post) if grad_norms_post else 0.0
            max_grad_pre = max(grad_norms_pre)
            # Log gradient norm summary for this epoch
            logger.info(f"[STABILITY] Epoch {epoch} grad_norm: avg_pre={avg_grad_pre:.4f}, max_pre={max_grad_pre:.4f}, avg_post={avg_grad_post:.4f}")
        else:
            avg_grad_pre = 0.0
            avg_grad_post = 0.0
            max_grad_pre = 0.0
            logger.warning(f"[STABILITY] Epoch {epoch}: No gradient norms collected (all batches skipped?)")
            
        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100. * correct / total,
            "gradient_norm": avg_grad_pre,
            "gradient_norm_max": max_grad_pre,
            "gradient_norm_post": avg_grad_post
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target.long())
            
            total_loss += loss.item()
            probs = F.softmax(output, dim=-1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.long()).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Yield to UI thread to prevent GUI freeze
            time.sleep(0)
            
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Class mapping: 0=SHORT, 1=HOLD, 2=LONG
        # Per-class metrics
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100. * correct / total,
        }
        
        # Calculate per-class precision, recall, F1
        class_names = ["short", "hold", "long"]
        for class_idx, class_name in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = ((all_preds == class_idx) & (all_targets == class_idx)).sum()
            fp = ((all_preds == class_idx) & (all_targets != class_idx)).sum()
            fn = ((all_preds != class_idx) & (all_targets == class_idx)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f"{class_name}_precision"] = precision * 100
            metrics[f"{class_name}_recall"] = recall * 100
            metrics[f"{class_name}_f1"] = f1 * 100
        
        # Directional accuracy (excluding HOLD predictions and targets)
        directional_mask = (all_targets != 1) & (all_preds != 1)
        if directional_mask.sum() > 0:
            directional_acc = (all_preds[directional_mask] == all_targets[directional_mask]).mean() * 100
        else:
            directional_acc = 0
        metrics["directional_acc"] = directional_acc
        
        # Macro F1 (average of per-class F1)
        macro_f1 = (metrics["short_f1"] + metrics["hold_f1"] + metrics["long_f1"]) / 3
        metrics["macro_f1"] = macro_f1
        
        # Class distribution (what is model predicting?)
        pred_short_pct = (all_preds == 0).mean() * 100
        pred_hold_pct = (all_preds == 1).mean() * 100
        pred_long_pct = (all_preds == 2).mean() * 100
        metrics["pred_short_pct"] = pred_short_pct
        metrics["pred_hold_pct"] = pred_hold_pct
        metrics["pred_long_pct"] = pred_long_pct
        
        # Backward compatibility
        metrics["long_precision"] = metrics["long_precision"]
        metrics["short_precision"] = metrics["short_precision"]
        
        return metrics
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        epochs = epochs or self.config.training.epochs
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "directional_acc": [], "macro_f1": [],
            "short_f1": [], "hold_f1": [], "long_f1": [],
            "gradient_norm": [], "gradient_norm_max": [], "gradient_norm_post": []
        }
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            for key, value in train_metrics.items():
                if key in history:
                    history[key].append(value)
            for key, value in val_metrics.items():
                if key in history:
                    history[key].append(value)
                    
            # Log all metrics to tensorboard
            self.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("val/acc", val_metrics["val_acc"], epoch)
            self.writer.add_scalar("val/directional_acc", val_metrics["directional_acc"], epoch)
            self.writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            self.writer.add_scalar("val/short_f1", val_metrics["short_f1"], epoch)
            self.writer.add_scalar("val/hold_f1", val_metrics["hold_f1"], epoch)
            self.writer.add_scalar("val/long_f1", val_metrics["long_f1"], epoch)
            
            # Only log in non-GUI mode (GUI has its own progress callback)
            if not self.gui_mode:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                    f"Dir Acc: {val_metrics['directional_acc']:.2f}%, "
                    f"Macro F1: {val_metrics['macro_f1']:.2f}%"
                )
                # Log class distribution every 10 epochs
                if epoch % 10 == 0:
                    logger.info(
                        f"  Class F1: SHORT={val_metrics['short_f1']:.1f}%, "
                        f"HOLD={val_metrics['hold_f1']:.1f}%, LONG={val_metrics['long_f1']:.1f}%"
                    )
                    logger.info(
                        f"  Pred Dist: SHORT={val_metrics['pred_short_pct']:.1f}%, "
                        f"HOLD={val_metrics['pred_hold_pct']:.1f}%, LONG={val_metrics['pred_long_pct']:.1f}%"
                    )
            
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint(f"best_{self.model.name}.pt")
            else:
                self.patience_counter += 1
            
            # Call epoch callback for GUI progress updates
            if self.epoch_callback is not None:
                # epoch_callback(epoch, train_loss, val_loss) -> returns continue_training bool
                should_continue = self.epoch_callback(
                    epoch - 1,  # 0-indexed for GUI compatibility
                    train_metrics["train_loss"],
                    val_metrics["val_loss"]
                )
                if not should_continue:
                    logger.info(f"Training stopped by callback at epoch {epoch}")
                    break
                
            if self.patience_counter >= self.config.training.patience:
                if not self.gui_mode:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                self.save_checkpoint(f"{self.model.name}_epoch_{epoch}.pt")
            
            # Yield to UI thread after each epoch to prevent GUI freeze
            time.sleep(0)
                
        self.writer.close()
        return history
    
    def save_checkpoint(self, filename: str):
        path = Path(self.config.training.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get FeatureEngineer version for tracking
        try:
            from data.pipeline import FeatureEngineer
            fe_version = FeatureEngineer.VERSION
        except:
            fe_version = "unknown"
        
        # Include model-specific config with input_dim for proper loading
        model_config = {
            "name": self.model.name,
            "input_dim": self.model.input_dim,
            "output_dim": self.model.output_dim,
        }
        
        # Try to capture model-specific attributes
        if hasattr(self.model, 'hidden_dim'):
            model_config["hidden_dim"] = self.model.hidden_dim
        if hasattr(self.model, 'd_model'):
            model_config["d_model"] = self.model.d_model
        if hasattr(self.model, 'sequence_length'):
            model_config["sequence_length"] = self.model.sequence_length
        if hasattr(self.model, 'channels'):
            model_config["channels"] = self.model.channels
        if hasattr(self.model, 'latent_dim'):
            model_config["latent_dim"] = self.model.latent_dim
        if hasattr(self.model, 'hidden_dims'):
            model_config["hidden_dims"] = self.model.hidden_dims
        if hasattr(self.model, 'num_assets'):
            model_config["num_assets"] = self.model.num_assets
        if hasattr(self.model, 'num_layers'):
            model_config["num_layers"] = self.model.num_layers
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
            "model_config": model_config,  # Model-specific config with input_dim
            "feature_engineer_version": fe_version,
            "training_mode": self.training_mode,  # Use actual training mode
            "horizon_periods": self.horizon_periods,  # Use actual horizon
        }
        
        # Include sklearn feature scaler if available (not AMP GradScaler)
        if self.feature_scaler is not None:
            try:
                checkpoint['scaler_mean'] = self.feature_scaler.mean_.tolist()
                checkpoint['scaler_scale'] = self.feature_scaler.scale_.tolist()
                checkpoint['scaler_var'] = self.feature_scaler.var_.tolist() if hasattr(self.feature_scaler, 'var_') else None
                checkpoint['scaler_n_features'] = self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else None
            except Exception as e:
                logger.warning(f"Could not save scaler state: {e}")
        
        # Include feature columns for validation at inference
        if self.feature_columns is not None:
            checkpoint['feature_columns'] = self.feature_columns
            
        torch.save(checkpoint, path)
        if not self.gui_mode:
            logger.info(f"Saved checkpoint to {path} (FE: {fe_version}, mode: {self.training_mode}, horizon: {self.horizon_periods})")
        
    def load_checkpoint(self, filename: str):
        path = Path(self.config.training.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.global_step = checkpoint["global_step"]
        
        logger.info(f"Loaded checkpoint from {path}")


class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        temperature: float = 0.07
    ):
        super().__init__(model, train_loader, val_loader, config, device)
        self.temperature = temperature
        
    def contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        
        mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        loss = -mean_log_prob.mean()
        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'get_embeddings'):
                embeddings = self.model.get_embeddings(data)
                output = self.model.classifier(embeddings)
            else:
                output = self.model(data)
                embeddings = output
                
            ce_loss = self.criterion(output, target.long())
            
            contrastive = self.contrastive_loss(embeddings, target)
            
            loss = ce_loss + 0.1 * contrastive
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_contrastive_loss += contrastive.item()
            
        n = len(self.train_loader)
        return {
            "train_loss": total_loss / n,
            "ce_loss": total_ce_loss / n,
            "contrastive_loss": total_contrastive_loss / n
        }


class CurriculumTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        difficulty_fn: Optional[Callable] = None
    ):
        super().__init__(model, train_loader, val_loader, config, device)
        self.difficulty_fn = difficulty_fn or self._default_difficulty
        self.current_difficulty = 0.0
        
    def _default_difficulty(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        volatility = data[:, :, 0].std(dim=1)
        return volatility
    
    def get_curriculum_weights(self, difficulties: torch.Tensor, epoch: int) -> torch.Tensor:
        progress = min(1.0, epoch / (self.config.training.epochs * 0.5))
        
        threshold = difficulties.quantile(progress)
        
        weights = (difficulties <= threshold).float()
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_weighted = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            data, target = data.to(self.device), target.to(self.device)
            
            difficulties = self.difficulty_fn(data, target)
            weights = self.get_curriculum_weights(difficulties, epoch)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            per_sample_loss = F.cross_entropy(output, target.long(), reduction='none')
            loss = (per_sample_loss * weights.to(self.device)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_weighted += weights.mean().item()
            
        return {
            "train_loss": total_loss / len(self.train_loader),
            "avg_curriculum_weight": total_weighted / len(self.train_loader)
        }


class OnlineTrainer:
    def __init__(
        self,
        model: nn.Module,
        config,
        device: str = "cuda",
        buffer_size: int = 1000
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate * 0.1
        )
        
        self.buffer = []
        self.buffer_size = buffer_size
        
        self.update_count = 0
        self.recent_losses = []
        
    def add_sample(self, features: np.ndarray, label: int):
        self.buffer.append((features, label))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def update(self, batch_size: int = 32) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
            
        self.model.train()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        features = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        labels = torch.LongTensor([b[1] for b in batch]).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(features)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        self.recent_losses.append(loss.item())
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
            
        return loss.item()
    
    def get_stats(self) -> Dict[str, float]:
        return {
            "buffer_size": len(self.buffer),
            "update_count": self.update_count,
            "avg_loss": np.mean(self.recent_losses) if self.recent_losses else 0
        }
