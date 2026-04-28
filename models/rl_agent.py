import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_encoder(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, values = self.forward(states)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class LSTMActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        self.encoder = nn.Linear(state_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def get_initial_hidden(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return h, c
    
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        if state.dim() == 2:
            state = state.unsqueeze(1)
            
        batch_size = state.size(0)
        device = state.device
        
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, device)
            
        encoded = self.encoder(state)
        lstm_out, new_hidden = self.lstm(encoded, hidden)
        
        features = lstm_out[:, -1, :]
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value, new_hidden


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer(capacity=10000)
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.policy.get_action(state_tensor)
        return action, log_prob, value
    
    def store_experience(self, experience: Experience):
        self.buffer.push(experience)
        
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return advantages, returns
    
    def update(self, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        if len(self.buffer) < batch_size:
            return {}
            
        experiences = list(self.buffer.buffer)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences]).to(self.device)
        
        rewards = [e.reward for e in experiences]
        values = [e.value for e in experiences]
        dones = [e.done for e in experiences]
        
        with torch.no_grad():
            _, next_value = self.policy(states[-1:])
            next_value = next_value.item()
            
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(experiences))
            
            for start in range(0, len(experiences), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                log_probs, values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
        self.buffer.clear()
        
        num_updates = epochs * (len(experiences) // batch_size + 1)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates
        }
    
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class TradingEnvironment:
    def __init__(
        self,
        data: np.ndarray,
        initial_balance: float = 10000.0,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        window_size: int = 100
    ):
        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        window = self.data[self.current_step - self.window_size:self.current_step]
        
        position_info = np.array([
            self.position / self.max_position,
            self.balance / self.initial_balance,
            self.total_pnl / self.initial_balance
        ])
        
        state = np.concatenate([window.flatten(), position_info])
        return state
    
    def _get_price(self) -> float:
        return self.data[self.current_step, 3]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        current_price = self._get_price()
        
        old_position = self.position
        
        if action == 0:
            if self.position <= 0:
                self.position = self.max_position
                self.entry_price = current_price
                cost = abs(self.position - old_position) * current_price * self.transaction_cost
                self.balance -= cost
        elif action == 1:
            if self.position >= 0:
                self.position = -self.max_position
                self.entry_price = current_price
                cost = abs(self.position - old_position) * current_price * self.transaction_cost
                self.balance -= cost
        else:
            if self.position != 0:
                pnl = self.position * (current_price - self.entry_price)
                cost = abs(self.position) * current_price * self.transaction_cost
                self.balance += pnl - cost
                self.total_pnl += pnl - cost
                self.trades.append({
                    "entry": self.entry_price,
                    "exit": current_price,
                    "position": self.position,
                    "pnl": pnl - cost
                })
                self.position = 0
                
        self.current_step += 1
        
        done = self.current_step >= len(self.data) - 1
        
        if self.position != 0:
            new_price = self.data[self.current_step, 3] if not done else current_price
            unrealized_pnl = self.position * (new_price - self.entry_price)
        else:
            unrealized_pnl = 0
            
        total_value = self.balance + unrealized_pnl
        returns = (total_value - self.initial_balance) / self.initial_balance
        
        if len(self.trades) > 1:
            trade_returns = [t["pnl"] / self.initial_balance for t in self.trades]
            sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
            
        reward = returns * 100 + sharpe * 0.1
        
        if self.balance < self.initial_balance * 0.5:
            reward -= 10
            done = True
            
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        
        info = {
            "balance": self.balance,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "num_trades": len(self.trades),
            "sharpe": sharpe
        }
        
        return next_state, reward, done, info
