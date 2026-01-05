# gnn_sac_formation_env.py
"""
GNN-SAC编队环境接口

特性：
1. 与现有ObstacleAvoidanceFormation环境集成
2. 构建双通道图观测（邻居/领航者分离）
3. 设计分布式多领航者奖励函数
4. 健康度评估与故障模拟
"""

import numpy as np
from typing import Dict, List, Tuple
from obstacle_environment_2d import ObstacleEnvironment2D


class GNNSACFormationEnv:
    """GNN-SAC编队环境"""
    
    def __init__(self,
                 num_agents: int = 5,
                 num_leaders: int = 2,
                 env_size: float = 300.0,
                 num_obstacles: int = 5,
                 max_neighbors: int = 4,
                 communication_range: float = 50.0,
                 dt: float = 0.1):
        """
        初始化环境
        
        Args:
            num_agents: 智能体总数
            num_leaders: 领航者数量（前num_leaders个作为领航者）
            max_neighbors: 最大邻居数
            communication_range: 通信范围
        """
        self.num_agents = num_agents
        self.num_leaders = num_leaders
        self.num_followers = num_agents - num_leaders
        self.env_size = env_size
        self.max_neighbors = max_neighbors
        self.communication_range = communication_range
        self.dt = dt
        
        # 创建障碍物环境
        self.obstacle_env = ObstacleEnvironment2D(
            env_size=env_size,
            num_obstacles=num_obstacles,
            seed=42
        )
        
        # 智能体状态
        self.agents = []
        
        # 目标位置
        self.start_pos = np.array([15.0, 15.0])
        self.goal_pos = np.array([285.0, 285.0])
        
        # 健康度（1.0=完全健康，0.0=失效）
        self.health_scores = np.ones(num_agents)
        
        # 编队配置
        self.formation_spacing = 8.0
        self.formation_offsets = self._generate_formation_offsets()
        
        # 奖励权重
        self.w_formation = 1.0      # 编队保持
        self.w_goal = 0.5           # 目标趋近
        self.w_collision = 2.0      # 碰撞惩罚
        self.w_consensus = 0.3      # 速度一致性
        self.w_energy = 0.05        # 能耗
        self.w_leader_coord = 0.4   # 领航者协调
        
        # 统计信息
        self.current_step = 0
        self.episode_reward = 0
        self.collision_count = 0
        
        print(f"✓ GNN-SAC编队环境初始化完成:")
        print(f"  智能体: {num_agents} (领航者: {num_leaders}, 跟随者: {self.num_followers})")
        print(f"  环境尺寸: {env_size}m × {env_size}m")
        print(f"  障碍物: {num_obstacles}个")
        print(f"  通信范围: {communication_range}m")
    
    def _generate_formation_offsets(self) -> np.ndarray:
        """生成编队偏移量（改进的V型编队）"""
        offsets = np.zeros((self.num_agents, 2))
        
        # 领航者：在前方形成一个小的领航者编队
        for i in range(self.num_leaders):
            if i == 0:
                # 中心领航者
                offsets[i] = np.array([0, 0])
            else:
                # 侧翼领航者
                side = 1 if i % 2 == 1 else -1
                position = (i + 1) // 2
                offsets[i] = np.array([
                    -position * self.formation_spacing * 0.3,
                    side * position * self.formation_spacing * 0.5
                ])
        
        # 跟随者：在领航者后方形成V型
        for i in range(self.num_leaders, self.num_agents):
            idx = i - self.num_leaders
            side = 1 if idx % 2 == 0 else -1
            position = (idx + 2) // 2
            
            offsets[i] = np.array([
                -position * self.formation_spacing * 0.8,
                side * position * self.formation_spacing * 0.7
            ])
        
        return offsets
    
    def reset(self) -> List[Dict[str, np.ndarray]]:
        """重置环境"""
        self.agents = []
        self.current_step = 0
        self.episode_reward = 0
        self.collision_count = 0
        self.health_scores = np.ones(self.num_agents)
        
        # 初始化智能体
        for i in range(self.num_agents):
            agent = {
                'id': i,
                'position': self.start_pos + self.formation_offsets[i],
                'velocity': np.zeros(2),
                'acceleration': np.zeros(2),
                'mass': 100.0,
                'max_thrust': 600.0 if i >= self.num_leaders else 500.0,
                'max_velocity': 3.5 if i >= self.num_leaders else 3.0,
                'role': 'leader' if i < self.num_leaders else 'follower',
                'health': 1.0
            }
            self.agents.append(agent)
        
        # 返回初始观测
        return self._get_observations()
    
    def _get_observations(self) -> List[Dict[str, np.ndarray]]:
        """
        构建双通道图观测
        
        Returns:
            observations: 每个智能体的观测列表
        """
        observations = []
        
        for i in range(self.num_agents):
            agent = self.agents[i]
            
            # 1. 自身特征 [pos(2), vel(2), acc(2), health(1), role(1)]
            self_feature = np.concatenate([
                agent['position'],
                agent['velocity'],
                agent['acceleration'],
                [agent['health']],
                [1.0 if agent['role'] == 'leader' else 0.0]
            ])
            
            # 2. 邻居通道：收集非领航者的邻居
            neighbor_features = []
            for j in range(self.num_agents):
                if j == i or self.agents[j]['role'] == 'leader':
                    continue
                
                dist = np.linalg.norm(agent['position'] - self.agents[j]['position'])
                if dist < self.communication_range:
                    # [relative_pos(2), relative_vel(2), health(1), distance(1)]
                    neighbor_feature = np.concatenate([
                        self.agents[j]['position'] - agent['position'],
                        self.agents[j]['velocity'] - agent['velocity'],
                        [self.agents[j]['health']],
                        [dist]
                    ])
                    neighbor_features.append(neighbor_feature)
            
            # 填充到固定长度
            while len(neighbor_features) < self.max_neighbors:
                neighbor_features.append(np.zeros(6))
            neighbor_features = neighbor_features[:self.max_neighbors]
            
            # 3. 领航者通道：收集所有领航者的信息
            leader_features = []
            for j in range(self.num_leaders):
                leader = self.agents[j]
                # [relative_pos(2), relative_vel(2), obstacle_dist(1)]
                obstacle_dist = self.obstacle_env.get_nearest_obstacle_distance(leader['position'])
                leader_feature = np.concatenate([
                    leader['position'] - agent['position'],
                    leader['velocity'] - agent['velocity'],
                    [obstacle_dist]
                ])
                leader_features.append(leader_feature)
            
            # 4. 构建掩码
            neighbor_mask = np.array([
                1.0 if i < len([n for n in neighbor_features if np.any(n != 0)]) else 0.0
                for i in range(self.max_neighbors)
            ])
            leader_mask = np.ones(self.num_leaders)
            
            # 5. 组装观测
            obs = {
                'self': self_feature,
                'neighbors': np.array(neighbor_features),
                'leaders': np.array(leader_features),
                'neighbor_mask': neighbor_mask,
                'leader_mask': leader_mask
            }
            
            observations.append(obs)
        
        return observations
    
    def step(self, actions: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray, Dict]:
        """
        执行环境步进
        
        Args:
            actions: [num_agents, 2] 所有智能体的动作
        
        Returns:
            observations: 新观测
            rewards: [num_agents] 奖励
            dones: [num_agents] 终止标志
            info: 额外信息
        """
        # 1. 应用动作（转换为控制力）
        for i, action in enumerate(actions):
            agent = self.agents[i]
            
            # 动作 ∈ [-1, 1] 映射到推力
            thrust = action * agent['max_thrust']
            
            # 计算加速度
            damping = 30.0
            damping_force = -damping * agent['velocity']
            total_force = thrust + damping_force
            acceleration = total_force / agent['mass']
            
            # 更新速度
            agent['velocity'] += acceleration * self.dt
            vel_norm = np.linalg.norm(agent['velocity'])
            if vel_norm > agent['max_velocity']:
                agent['velocity'] = agent['velocity'] * (agent['max_velocity'] / vel_norm)
            
            # 更新位置
            agent['position'] += agent['velocity'] * self.dt
            agent['position'] = np.clip(agent['position'], 0, self.env_size)
            
            # 记录加速度
            agent['acceleration'] = acceleration
        
        # 2. 计算奖励
        rewards = self._compute_rewards()
        
        # 3. 检查终止条件
        dones = self._check_done()
        
        # 4. 更新健康度（模拟故障）
        self._update_health()
        
        # 5. 获取新观测
        observations = self._get_observations()
        
        # 6. 统计信息
        self.current_step += 1
        self.episode_reward += np.sum(rewards)
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'collision_count': self.collision_count,
            'mean_health': np.mean(self.health_scores),
            'formation_error': self._calculate_formation_error(),
            'goal_distance': np.linalg.norm(self._get_centroid() - self.goal_pos)
        }
        
        return observations, rewards, dones, info
    
    def _compute_rewards(self) -> np.ndarray:
        """
        计算改进的奖励函数
        
        Returns:
            rewards: [num_agents] 每个智能体的奖励
        """
        rewards = np.zeros(self.num_agents)
        
        # 计算编队质心
        centroid = self._get_centroid()
        
        # 领航者质心（用于协调奖励）
        leader_centroid = np.mean([self.agents[i]['position'] for i in range(self.num_leaders)], axis=0)
        
        for i in range(self.num_agents):
            agent = self.agents[i]
            
            # 1. 编队保持奖励（相对于期望偏移）
            if agent['role'] == 'leader':
                # 领航者：保持在领航者编队中
                desired_pos = leader_centroid + self.formation_offsets[i]
            else:
                # 跟随者：保持在全局编队中
                desired_pos = centroid + self.formation_offsets[i]
            
            formation_error = np.linalg.norm(agent['position'] - desired_pos)
            r_formation = -self.w_formation * (formation_error ** 2)
            
            # 2. 目标趋近奖励（仅领航者）
            if agent['role'] == 'leader':
                goal_dist = np.linalg.norm(agent['position'] - self.goal_pos)
                r_goal = -self.w_goal * goal_dist
            else:
                r_goal = 0
            
            # 3. 碰撞惩罚
            if self.obstacle_env.check_collision(agent['position'], agent_radius=1.5):
                r_collision = -self.w_collision * 100
                self.collision_count += 1
            else:
                # 接近障碍物也有小惩罚
                nearest_dist = self.obstacle_env.get_nearest_obstacle_distance(agent['position'])
                if nearest_dist < 5.0:
                    r_collision = -self.w_collision * (5.0 - nearest_dist)
                else:
                    r_collision = 0
            
            # 4. 速度一致性奖励（与邻居）
            neighbors = self._get_neighbors(i)
            if len(neighbors) > 0:
                vel_diff = sum([
                    np.linalg.norm(agent['velocity'] - self.agents[j]['velocity']) ** 2
                    for j in neighbors
                ]) / len(neighbors)
                r_consensus = -self.w_consensus * vel_diff
            else:
                r_consensus = 0
            
            # 5. 能耗惩罚
            r_energy = -self.w_energy * np.linalg.norm(agent['acceleration']) ** 2
            
            # 6. 领航者协调奖励（仅领航者）
            if agent['role'] == 'leader':
                # 领航者之间保持适当距离
                leader_dists = []
                for j in range(self.num_leaders):
                    if j != i:
                        dist = np.linalg.norm(agent['position'] - self.agents[j]['position'])
                        leader_dists.append(dist)
                
                if len(leader_dists) > 0:
                    mean_dist = np.mean(leader_dists)
                    desired_dist = self.formation_spacing * 0.5
                    r_leader_coord = -self.w_leader_coord * (mean_dist - desired_dist) ** 2
                else:
                    r_leader_coord = 0
            else:
                r_leader_coord = 0
            
            # 总奖励
            total_reward = (
                r_formation + r_goal + r_collision + 
                r_consensus + r_energy + r_leader_coord
            )
            
            # 健康度衰减
            total_reward *= agent['health']
            
            rewards[i] = total_reward
        
        return rewards
    
    def _get_neighbors(self, agent_id: int) -> List[int]:
        """获取智能体的邻居"""
        neighbors = []
        agent_pos = self.agents[agent_id]['position']
        
        for j in range(self.num_agents):
            if j != agent_id:
                dist = np.linalg.norm(agent_pos - self.agents[j]['position'])
                if dist < self.communication_range:
                    neighbors.append(j)
        
        return neighbors
    
    def _get_centroid(self) -> np.ndarray:
        """计算编队质心"""
        positions = [agent['position'] for agent in self.agents]
        return np.mean(positions, axis=0)
    
    def _calculate_formation_error(self) -> float:
        """计算编队误差"""
        centroid = self._get_centroid()
        total_error = 0
        
        for i, agent in enumerate(self.agents):
            desired_pos = centroid + self.formation_offsets[i]
            error = np.linalg.norm(agent['position'] - desired_pos)
            total_error += error
        
        return total_error / self.num_agents
    
    def _check_done(self) -> np.ndarray:
        """检查终止条件"""
        dones = np.zeros(self.num_agents, dtype=bool)
        
        # 达到目标
        centroid = self._get_centroid()
        if np.linalg.norm(centroid - self.goal_pos) < 10.0:
            dones[:] = True
        
        # 超时
        if self.current_step >= 2000:
            dones[:] = True
        
        # 领航者全部失效
        leader_health = np.mean([self.agents[i]['health'] for i in range(self.num_leaders)])
        if leader_health < 0.3:
            dones[:] = True
        
        return dones
    
    def _update_health(self):
        """更新健康度（模拟故障）"""
        # 随机故障（低概率）
        for i in range(self.num_agents):
            if np.random.random() < 0.0001:  # 0.01%概率
                self.health_scores[i] *= 0.9
            
            # 碰撞降低健康度
            if self.obstacle_env.check_collision(self.agents[i]['position'], 1.5):
                self.health_scores[i] *= 0.95
            
            # 更新智能体健康度
            self.agents[i]['health'] = self.health_scores[i]


if __name__ == "__main__":
    print("=" * 80)
    print("GNN-SAC编队环境测试")
    print("=" * 80)
    
    # 创建环境
    env = GNNSACFormationEnv(
        num_agents=5,
        num_leaders=2,
        env_size=300.0,
        num_obstacles=5
    )
    
    # 重置环境
    obs_list = env.reset()
    
    print(f"\n初始观测:")
    for i, obs in enumerate(obs_list):
        print(f"  智能体{i}:")
        print(f"    自身特征形状: {obs['self'].shape}")
        print(f"    邻居特征形状: {obs['neighbors'].shape}")
        print(f"    领航者特征形状: {obs['leaders'].shape}")
        print(f"    邻居掩码: {obs['neighbor_mask']}")
    
    # 测试步进
    actions = np.random.randn(env.num_agents, 2) * 0.1
    obs_list, rewards, dones, info = env.step(actions)
    
    print(f"\n步进测试:")
    print(f"  奖励: {rewards}")
    print(f"  终止: {dones}")
    print(f"  信息: {info}")
    
    print("\n✓ 环境测试通过!")