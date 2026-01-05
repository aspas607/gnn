# gnn_sac_dual_channel.py
"""
GNN-SAC双通道聚合编队算法 - 核心网络架构

创新点：
1. 双通道图聚合：邻居通道 + 领航者通道分离
2. SAC算法：熵正则化 + 双Q网络
3. 动态拓扑感知：自适应注意力机制
4. 分布式多领航者：3个领航者协同决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional


class DualChannelGraphAggregator(nn.Module):
    """双通道图聚合器 - 核心创新"""
    
    def __init__(self, 
                 self_dim: int = 8,      # 自身特征维度
                 neighbor_dim: int = 6,   # 邻居特征维度
                 leader_dim: int = 5,     # 领航者特征维度
                 hidden_dim: int = 128):
        """
        Args:
            self_dim: 自身状态维度 [pos(2), vel(2), acc(2), health(1), role(1)]
            neighbor_dim: 邻居特征维度 [relative_pos(2), relative_vel(2), health(1), distance(1)]
            leader_dim: 领航者特征维度 [relative_pos(2), relative_vel(2), obstacle_dist(1)]
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.self_dim = self_dim
        self.neighbor_dim = neighbor_dim
        self.leader_dim = leader_dim
        self.hidden_dim = hidden_dim
        
        # 自身特征编码器
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 邻居通道编码器
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 领航者通道编码器
        self.leader_encoder = nn.Sequential(
            nn.Linear(leader_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 邻居通道注意力聚合
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 领航者通道注意力聚合
        self.leader_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 门控融合机制（决定三个通道的权重）
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, 
                self_features: torch.Tensor,       # [batch, self_dim]
                neighbor_features: torch.Tensor,   # [batch, max_neighbors, neighbor_dim]
                leader_features: torch.Tensor,     # [batch, num_leaders, leader_dim]
                neighbor_mask: torch.Tensor = None, # [batch, max_neighbors]
                leader_mask: torch.Tensor = None    # [batch, num_leaders]
               ) -> torch.Tensor:
        """
        双通道图聚合前向传播
        
        Returns:
            aggregated_features: [batch, hidden_dim] 聚合后的特征
        """
        batch_size = self_features.size(0)
        
        # 1. 编码自身特征
        self_encoded = self.self_encoder(self_features)  # [batch, hidden]
        
        # 2. 邻居通道聚合
        if neighbor_features.size(1) > 0:
            # 编码邻居特征
            batch_size, num_neighbors, _ = neighbor_features.shape
            neighbor_encoded = self.neighbor_encoder(
                neighbor_features.reshape(-1, self.neighbor_dim)
            ).reshape(batch_size, num_neighbors, self.hidden_dim)
            
            # 注意力聚合（query=自身，key/value=邻居）
            self_query = self_encoded.unsqueeze(1)  # [batch, 1, hidden]
            
            # 创建注意力掩码
            if neighbor_mask is not None:
                attn_mask = ~neighbor_mask.bool()  # True表示需要mask的位置
            else:
                attn_mask = None
            
            neighbor_aggregated, _ = self.neighbor_attention(
                query=self_query,
                key=neighbor_encoded,
                value=neighbor_encoded,
                key_padding_mask=attn_mask
            )
            neighbor_aggregated = neighbor_aggregated.squeeze(1)  # [batch, hidden]
        else:
            neighbor_aggregated = torch.zeros_like(self_encoded)
        
        # 3. 领航者通道聚合
        if leader_features.size(1) > 0:
            # 编码领航者特征
            batch_size, num_leaders, _ = leader_features.shape
            leader_encoded = self.leader_encoder(
                leader_features.reshape(-1, self.leader_dim)
            ).reshape(batch_size, num_leaders, self.hidden_dim)
            
            # 注意力聚合
            self_query = self_encoded.unsqueeze(1)
            
            if leader_mask is not None:
                attn_mask = ~leader_mask.bool()
            else:
                attn_mask = None
            
            leader_aggregated, _ = self.leader_attention(
                query=self_query,
                key=leader_encoded,
                value=leader_encoded,
                key_padding_mask=attn_mask
            )
            leader_aggregated = leader_aggregated.squeeze(1)  # [batch, hidden]
        else:
            leader_aggregated = torch.zeros_like(self_encoded)
        
        # 4. 门控融合
        # 连接三个通道
        concat_features = torch.cat([
            self_encoded, 
            neighbor_aggregated, 
            leader_aggregated
        ], dim=-1)  # [batch, hidden*3]
        
        # 计算门控权重
        gate_weights = self.gate_network(concat_features)  # [batch, 3]
        
        # 加权融合
        weighted_features = (
            gate_weights[:, 0:1] * self_encoded +
            gate_weights[:, 1:2] * neighbor_aggregated +
            gate_weights[:, 2:3] * leader_aggregated
        )
        
        # 5. 最终融合
        final_features = self.fusion_layer(concat_features)
        
        return final_features, gate_weights


class GNNSACActor(nn.Module):
    """GNN-SAC策略网络（Actor）"""
    
    def __init__(self,
                 self_dim: int = 8,
                 neighbor_dim: int = 6,
                 leader_dim: int = 5,
                 hidden_dim: int = 128,
                 action_dim: int = 2):
        super().__init__()
        
        # 双通道图聚合器
        self.dual_aggregator = DualChannelGraphAggregator(
            self_dim=self_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim
        )
        
        # 策略头（输出动作分布的均值）
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 标准差头（输出动作分布的标准差）
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 标准差范围
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs_dict: 包含 'self', 'neighbors', 'leaders', 'neighbor_mask', 'leader_mask'
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        # 双通道聚合
        features, gate_weights = self.dual_aggregator(
            self_features=obs_dict['self'],
            neighbor_features=obs_dict['neighbors'],
            leader_features=obs_dict['leaders'],
            neighbor_mask=obs_dict.get('neighbor_mask'),
            leader_mask=obs_dict.get('leader_mask')
        )
        
        # 计算动作分布参数
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # 采样动作（重参数化技巧）
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 使用rsample以支持梯度反向传播
        
        # Tanh压缩到[-1, 1]
        action = torch.tanh(x_t)
        
        # 计算对数概率（考虑Tanh变换）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, gate_weights
    
    def get_action(self, obs_dict: Dict[str, torch.Tensor], deterministic: bool = False):
        """获取动作（用于测试）"""
        with torch.no_grad():
            features, gate_weights = self.dual_aggregator(
                self_features=obs_dict['self'],
                neighbor_features=obs_dict['neighbors'],
                leader_features=obs_dict['leaders'],
                neighbor_mask=obs_dict.get('neighbor_mask'),
                leader_mask=obs_dict.get('leader_mask')
            )
            
            mean = self.mean_head(features)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                log_std = self.log_std_head(features)
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)
                normal = Normal(mean, std)
                x_t = normal.sample()
                action = torch.tanh(x_t)
            
            return action, gate_weights


class GNNSACCritic(nn.Module):
    """GNN-SAC价值网络（Critic）- 双Q网络"""
    
    def __init__(self,
                 self_dim: int = 8,
                 neighbor_dim: int = 6,
                 leader_dim: int = 5,
                 hidden_dim: int = 128,
                 action_dim: int = 2):
        super().__init__()
        
        # Q1网络
        self.q1_aggregator = DualChannelGraphAggregator(
            self_dim=self_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim
        )
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Q2网络
        self.q2_aggregator = DualChannelGraphAggregator(
            self_dim=self_dim,
            neighbor_dim=neighbor_dim,
            leader_dim=leader_dim,
            hidden_dim=hidden_dim
        )
        self.q2_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor):
        """计算Q值"""
        # Q1
        features1, _ = self.q1_aggregator(
            self_features=obs_dict['self'],
            neighbor_features=obs_dict['neighbors'],
            leader_features=obs_dict['leaders'],
            neighbor_mask=obs_dict.get('neighbor_mask'),
            leader_mask=obs_dict.get('leader_mask')
        )
        q1 = self.q1_head(torch.cat([features1, action], dim=-1))
        
        # Q2
        features2, _ = self.q2_aggregator(
            self_features=obs_dict['self'],
            neighbor_features=obs_dict['neighbors'],
            leader_features=obs_dict['leaders'],
            neighbor_mask=obs_dict.get('neighbor_mask'),
            leader_mask=obs_dict.get('leader_mask')
        )
        q2 = self.q2_head(torch.cat([features2, action], dim=-1))
        
        return q1, q2
    
    def q1_forward(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor):
        """只计算Q1（用于策略更新）"""
        features1, _ = self.q1_aggregator(
            self_features=obs_dict['self'],
            neighbor_features=obs_dict['neighbors'],
            leader_features=obs_dict['leaders'],
            neighbor_mask=obs_dict.get('neighbor_mask'),
            leader_mask=obs_dict.get('leader_mask')
        )
        q1 = self.q1_head(torch.cat([features1, action], dim=-1))
        return q1


class AlphaNetwork(nn.Module):
    """自动调节温度参数α的网络"""
    
    def __init__(self, init_alpha: float = 0.2):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha)))
    
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)


if __name__ == "__main__":
    print("=" * 80)
    print("GNN-SAC双通道聚合编队算法 - 网络架构")
    print("=" * 80)
    print("\n核心创新：")
    print("1. ✓ 双通道图聚合：邻居通道 + 领航者通道分离")
    print("2. ✓ 门控融合机制：自适应调节三个通道权重")
    print("3. ✓ SAC算法：熵正则化 + 双Q网络")
    print("4. ✓ 动态拓扑感知：多头注意力机制")
    print("\n网络维度：")
    print("  自身特征: [pos(2), vel(2), acc(2), health(1), role(1)] = 8")
    print("  邻居特征: [relative_pos(2), relative_vel(2), health(1), dist(1)] = 6")
    print("  领航者特征: [relative_pos(2), relative_vel(2), obs_dist(1)] = 5")
    print("  动作空间: [thrust_x, thrust_y] ∈ [-1, 1]²")
    print("=" * 80)
    
    # 测试网络
    batch_size = 4
    max_neighbors = 5
    num_leaders = 3
    
    # 构造测试数据
    obs = {
        'self': torch.randn(batch_size, 8),
        'neighbors': torch.randn(batch_size, max_neighbors, 6),
        'leaders': torch.randn(batch_size, num_leaders, 5),
        'neighbor_mask': torch.ones(batch_size, max_neighbors).bool(),
        'leader_mask': torch.ones(batch_size, num_leaders).bool()
    }
    
    # 测试Actor
    actor = GNNSACActor()
    action, log_prob, gate_weights = actor(obs)
    print(f"\n测试Actor:")
    print(f"  动作形状: {action.shape}")
    print(f"  对数概率形状: {log_prob.shape}")
    print(f"  门控权重形状: {gate_weights.shape}")
    print(f"  门控权重示例: {gate_weights[0]}")
    
    # 测试Critic
    critic = GNNSACCritic()
    q1, q2 = critic(obs, action)
    print(f"\n测试Critic:")
    print(f"  Q1形状: {q1.shape}")
    print(f"  Q2形状: {q2.shape}")
    
    print("\n✓ 网络测试通过!")