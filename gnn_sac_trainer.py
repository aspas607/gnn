# gnn_sac_trainer.py
"""
GNN-SAC训练器 - Soft Actor-Critic算法实现

特性：
1. 双Q网络 + 目标网络软更新
2. 熵正则化 + 自动调节温度参数
3. 优先级经验回放
4. 分布式多领航者协同训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple
from gnn_sac_dual_channel import GNNSACActor, GNNSACCritic, AlphaNetwork


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀采样，1=完全优先级）
            beta: 重要性采样权重（0.4→1.0 annealing）
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # 每次采样时增加beta
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, td_error: float = 1.0):
        """添加经验"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # 新样本赋予最大优先级
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """采样批次"""
        if self.size < batch_size:
            batch_size = self.size
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        
        # 提取批次
        batch = [self.buffer[idx] for idx in indices]
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # 避免零优先级
    
    def __len__(self):
        return self.size


class GNNSACTrainer:
    """GNN-SAC训练器"""
    
    def __init__(self,
                 num_agents: int = 3,
                 num_leaders: int = 2,
                 state_dim: int = 8,
                 neighbor_dim: int = 6,
                 leader_dim: int = 5,
                 action_dim: int = 2,
                 hidden_dim: int = 128,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 target_entropy: float = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            num_agents: 智能体数量
            num_leaders: 领航者数量
            auto_alpha: 是否自动调节温度参数
            target_entropy: 目标熵（None则自动设置为-action_dim）
        """
        self.num_agents = num_agents
        self.num_leaders = num_leaders
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # 创建网络
        self.actor = GNNSACActor(
            self_dim=state_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)
        
        self.critic = GNNSACCritic(
            self_dim=state_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)
        
        # 目标Critic网络
        self.critic_target = GNNSACCritic(
            self_dim=state_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 温度参数
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy is None:
                self.target_entropy = -action_dim  # 常用启发式
            else:
                self.target_entropy = target_entropy
            
            self.alpha_network = AlphaNetwork(init_alpha=alpha).to(device)
            self.alpha_optimizer = optim.Adam([self.alpha_network.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
        # 统计信息
        self.train_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.alpha_values = []
        self.gate_weight_history = []  # 记录门控权重
    
    def select_action(self, obs_dict: Dict[str, np.ndarray], deterministic: bool = False):
        """
        选择动作
        
        Args:
            obs_dict: 观测字典（numpy格式）
            deterministic: 是否使用确定性策略
        
        Returns:
            action: numpy数组
        """
        # 转换为tensor
        obs_tensor = self._to_tensor(obs_dict)
        
        with torch.no_grad():
            action, gate_weights = self.actor.get_action(obs_tensor, deterministic)
            
            # 记录门控权重（用于分析）
            self.gate_weight_history.append(gate_weights.cpu().numpy())
        
        return action.cpu().numpy(), gate_weights.cpu().numpy()
    
    def train_step_sac(self, batch_size: int = 256):
        """执行一步SAC训练"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样批次
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        
        # 解包批次
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states_tensor = self._batch_to_tensor(states)
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = self._batch_to_tensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # 获取当前alpha
        if self.auto_alpha:
            alpha = self.alpha_network.alpha
        else:
            alpha = self.alpha
        
        # ==================== 更新Critic ====================
        with torch.no_grad():
            # 下一个状态的动作和对数概率
            next_action, next_log_prob, _ = self.actor(next_states_tensor)
            
            # 目标Q值（双Q取最小）
            next_q1_target, next_q2_target = self.critic_target(next_states_tensor, next_action)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # TD目标（SAC特有：Q值 - α*log_prob）
            td_target = rewards_tensor + self.gamma * (1 - dones_tensor) * (
                next_q_target - alpha * next_log_prob
            )
        
        # 当前Q值
        q1, q2 = self.critic(states_tensor, actions_tensor)
        
        # Critic损失（加权MSE）
        critic_loss = (
            weights_tensor * (q1 - td_target).pow(2) +
            weights_tensor * (q2 - td_target).pow(2)
        ).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = (q1 - td_target).abs().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # ==================== 更新Actor ====================
        # 重新采样动作（用于策略梯度）
        new_action, log_prob, gate_weights = self.actor(states_tensor)
        
        # 计算Q值（只用Q1）
        q1_new = self.critic.q1_forward(states_tensor, new_action)
        
        # Actor损失（最大化 Q - α*log_prob）
        actor_loss = (alpha * log_prob - q1_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ==================== 更新Alpha（如果自动调节） ====================
        if self.auto_alpha:
            alpha_loss = -(self.alpha_network.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha_losses.append(alpha_loss.item())
        
        # ==================== 软更新目标网络 ====================
        self._soft_update(self.critic, self.critic_target, self.tau)
        
        # 记录统计信息
        self.train_step += 1
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        if self.auto_alpha:
            self.alpha_values.append(alpha.item())
        
        # 返回训练统计
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': alpha.item() if self.auto_alpha else alpha,
            'alpha_loss': alpha_loss.item() if self.auto_alpha else 0.0,
            'mean_gate_weights': gate_weights.mean(dim=0).cpu().numpy()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def _to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """numpy字典转tensor字典"""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }
    
    def _batch_to_tensor(self, batch_obs: List[Dict]) -> Dict[str, torch.Tensor]:
        """批量观测转tensor"""
        # 收集所有键的值
        keys = batch_obs[0].keys()
        tensor_dict = {}
        
        for key in keys:
            values = [obs[key] for obs in batch_obs]
            tensor_dict[key] = torch.FloatTensor(np.array(values)).to(self.device)
        
        return tensor_dict
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'train_step': self.train_step
        }
        
        if self.auto_alpha:
            checkpoint['alpha_network'] = self.alpha_network.state_dict()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.train_step = checkpoint['train_step']
        
        if self.auto_alpha and 'alpha_network' in checkpoint:
            self.alpha_network.load_state_dict(checkpoint['alpha_network'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        print(f"✓ 检查点已加载: {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("GNN-SAC训练器测试")
    print("=" * 80)
    
    # 创建训练器
    trainer = GNNSACTrainer(
        num_agents=3,
        num_leaders=2,
        auto_alpha=True
    )
    
    print(f"\n训练器配置:")
    print(f"  智能体数量: {trainer.num_agents}")
    print(f"  领航者数量: {trainer.num_leaders}")
    print(f"  自动调节α: {trainer.auto_alpha}")
    print(f"  设备: {trainer.device}")
    
    # 测试动作选择
    test_obs = {
        'self': np.random.randn(8),
        'neighbors': np.random.randn(5, 6),
        'leaders': np.random.randn(2, 5),
        'neighbor_mask': np.ones(5),
        'leader_mask': np.ones(2)
    }
    
    action, gate_weights = trainer.select_action(test_obs, deterministic=False)
    print(f"\n测试动作选择:")
    print(f"  动作: {action}")
    print(f"  门控权重: {gate_weights}")
    
    print("\n✓ 训练器测试通过!")