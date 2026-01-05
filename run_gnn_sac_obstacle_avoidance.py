# run_gnn_sac_obstacle_avoidance.py
"""
GNN-SAC编队避障 - 与现有代码框架集成

特性：
1. 复用obstacle_environment_2d障碍物环境
2. 使用GNN-SAC双通道聚合算法
3. 分布式多领航者协同控制
4. 完整可视化（整体图 + 放大图 + 门控权重）

使用方法:
python run_gnn_sac_obstacle_avoidance.py --mode train  # 训练模式
python run_gnn_sac_obstacle_avoidance.py --mode test --checkpoint checkpoints/model.pth  # 测试模式
"""

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

from obstacle_environment_2d import ObstacleEnvironment2D
from gnn_sac_formation_env import GNNSACFormationEnv
from gnn_sac_trainer import GNNSACTrainer


class GNNSACObstacleAvoidanceFormation:
    """GNN-SAC障碍物避障编队系统（集成版）"""
    
    def __init__(self,
                 num_auv: int = 5,
                 num_leaders: int = 2,
                 env_size: float = 300.0,
                 checkpoint_path: str = None):
        """
        初始化系统
        
        Args:
            num_auv: AUV总数
            num_leaders: 领航者数量
            env_size: 环境尺寸
            checkpoint_path: 预训练模型路径（None则使用未训练模型）
        """
        self.num_auv = num_auv
        self.num_leaders = num_leaders
        self.env_size = env_size
        
        print("=" * 80)
        print("GNN-SAC障碍物避障编队系统")
        print("=" * 80)
        print(f"智能体: {num_auv} (领航者: {num_leaders}, 跟随者: {num_auv - num_leaders})")
        print(f"环境: {env_size}m × {env_size}m, 5个障碍物")
        print("算法: GNN-SAC双通道聚合")
        print("=" * 80 + "\n")
        
        # 创建环境
        self.env = GNNSACFormationEnv(
            num_agents=num_auv,
            num_leaders=num_leaders,
            env_size=env_size,
            num_obstacles=5
        )
        
        # 创建训练器
        self.trainer = GNNSACTrainer(
            num_agents=num_auv,
            num_leaders=num_leaders,
            state_dim=8,
            neighbor_dim=6,
            leader_dim=5,
            action_dim=2,
            hidden_dim=128
        )
        
        # 加载预训练模型
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
            self.trainer.actor.eval()
            print(f"✓ 已加载预训练模型: {checkpoint_path}\n")
        else:
            print("⚠ 未加载预训练模型，将使用随机初始化的策略\n")
        
        # 可视化设置
        self.fig = None
        self.ax_overall = None
        self.ax_zoom = None
        self.ax_gates = None
        
        # 颜色方案
        self.auv_colors = ['#FF0000', '#45B7D1', '#4ECDC4', '#96CEB4', '#FFEAA7']
        
        # 历史记录
        self.positions_history = []
        self.gate_weights_history = []
    
    def run_experiment(self, 
                      max_steps: int = 2000,
                      visualize: bool = True,
                      deterministic: bool = False):
        """
        运行实验
        
        Args:
            max_steps: 最大步数
            visualize: 是否可视化
            deterministic: 是否使用确定性策略
        """
        print("开始运行实验...\n")
        
        if visualize:
            self._setup_visualization()
        
        # 重置环境
        obs_list = self.env.reset()
        self.positions_history = []
        self.gate_weights_history = []
        
        episode_reward = 0
        start_time = time.time()
        
        for step in range(max_steps):
            # 选择动作
            actions = []
            gate_weights_batch = []
            
            for obs in obs_list:
                action, gate_w = self.trainer.select_action(obs, deterministic=deterministic)
                actions.append(action[0])
                gate_weights_batch.append(gate_w[0])
            
            actions = np.array(actions)
            gate_weights = np.array(gate_weights_batch)
            
            # 环境步进
            next_obs_list, rewards, dones, info = self.env.step(actions)
            
            # 记录历史
            positions = np.array([agent['position'] for agent in self.env.agents])
            self.positions_history.append(positions.copy())
            self.gate_weights_history.append(gate_weights.copy())
            
            # 更新
            obs_list = next_obs_list
            episode_reward += np.sum(rewards)
            
            # 可视化
            if visualize and step % 5 == 0:
                self._update_visualization(positions, gate_weights)
            
            # 进度显示
            if step % 100 == 0:
                print(f"步骤 {step}/{max_steps}: "
                      f"奖励={np.sum(rewards):.2f}, "
                      f"编队误差={info['formation_error']:.2f}m, "
                      f"目标距离={info['goal_distance']:.2f}m")
            
            # 检查终止
            if np.all(dones):
                print(f"\n✓ 到达目标，在步骤 {step} 完成")
                break
        
        elapsed_time = time.time() - start_time
        self._print_summary(step, episode_reward, info, elapsed_time)
        
        if visualize:
            self._save_final_visualization()
    
    def _setup_visualization(self):
        """设置可视化（三个子图）"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        self.fig = plt.figure(figsize=(24, 10))
        
        # (a) 整体图
        self.ax_overall = self.fig.add_subplot(131)
        self.ax_overall.set_xlim(0, self.env_size)
        self.ax_overall.set_ylim(0, self.env_size)
        self.ax_overall.set_xlabel('X Location (m)', fontsize=13)
        self.ax_overall.set_ylabel('Y Location (m)', fontsize=13)
        self.ax_overall.set_title('(a) Overall View - GNN-SAC', fontsize=14, fontweight='bold')
        self.ax_overall.set_aspect('equal')
        self.ax_overall.grid(True, alpha=0.3)
        
        # 绘制障碍物
        self.env.obstacle_env.visualize(self.ax_overall, show_detection_range=False)
        
        # 标记起点和终点
        self.ax_overall.plot(self.env.start_pos[0], self.env.start_pos[1], 'ro',
                            markersize=12, label='Start', zorder=20)
        self.ax_overall.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'g*',
                            markersize=18, label='Goal', zorder=20)
        
        self.ax_overall.legend(fontsize=11)
        
        # (b) 放大图
        self.ax_zoom = self.fig.add_subplot(132)
        self.ax_zoom.set_xlabel('X Location (m)', fontsize=13)
        self.ax_zoom.set_ylabel('Y Location (m)', fontsize=13)
        self.ax_zoom.set_title('(b) Zoomed View', fontsize=14, fontweight='bold')
        self.ax_zoom.set_aspect('equal')
        self.ax_zoom.grid(True, alpha=0.3)
        
        # (c) 门控权重图
        self.ax_gates = self.fig.add_subplot(133)
        self.ax_gates.set_xlabel('Steps', fontsize=13)
        self.ax_gates.set_ylabel('Gate Weight', fontsize=13)
        self.ax_gates.set_title('(c) Dual-Channel Gate Weights', fontsize=14, fontweight='bold')
        self.ax_gates.grid(True, alpha=0.3)
        self.ax_gates.set_ylim([0, 1])
        
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_visualization(self, positions, gate_weights):
        """更新可视化"""
        # 清除动态元素
        for artist in list(self.ax_overall.lines[2:]) + list(self.ax_overall.collections[len(self.env.obstacle_env.obstacles):]):
            artist.remove()
        
        for artist in list(self.ax_zoom.collections) + list(self.ax_zoom.patches) + list(self.ax_zoom.lines):
            artist.remove()
        
        # ========== 整体图：绘制轨迹 ==========
        if len(self.positions_history) > 1:
            for i in range(self.num_auv):
                color = self.auv_colors[i % len(self.auv_colors)]
                hist = np.array([h[i] for h in self.positions_history])
                label = f'Leader-{i}' if i < self.num_leaders else f'Follower-{i - self.num_leaders}'
                self.ax_overall.plot(hist[:, 0], hist[:, 1],
                                    color=color, linewidth=2.5, alpha=0.8,
                                    label=label if self.env.current_step < 10 else '')
        
        # 绘制当前位置
        for i, pos in enumerate(positions):
            color = self.auv_colors[i % len(self.auv_colors)]
            size = 150 if i < self.num_leaders else 100
            marker = 's' if i < self.num_leaders else 'o'
            self.ax_overall.scatter(pos[0], pos[1], color=color, s=size,
                                   marker=marker, edgecolors='black',
                                   linewidth=2, zorder=15)
        
        # ========== 放大图：显示局部细节 ==========
        # 以编队质心为中心
        centroid = np.mean(positions, axis=0)
        zoom_range = 50.0
        
        self.ax_zoom.set_xlim(centroid[0] - zoom_range, centroid[0] + zoom_range)
        self.ax_zoom.set_ylim(centroid[1] - zoom_range, centroid[1] + zoom_range)
        
        # 绘制可见的障碍物
        for obstacle in self.env.obstacle_env.obstacles:
            if np.linalg.norm(obstacle.center - centroid) < zoom_range + obstacle.radius:
                circle = Circle(obstacle.center, obstacle.radius,
                              color='cornflowerblue', alpha=0.7,
                              edgecolor='darkblue', linewidth=2, zorder=5)
                self.ax_zoom.add_patch(circle)
        
        # 绘制轨迹（最近100步）
        if len(self.positions_history) > 1:
            start_idx = max(0, len(self.positions_history) - 100)
            for i in range(self.num_auv):
                color = self.auv_colors[i % len(self.auv_colors)]
                hist = np.array([h[i] for h in self.positions_history[start_idx:]])
                self.ax_zoom.plot(hist[:, 0], hist[:, 1],
                                color=color, linewidth=2.5, alpha=0.8)
        
        # 绘制当前位置和编队连接线
        for i, pos in enumerate(positions):
            color = self.auv_colors[i % len(self.auv_colors)]
            size = 120 if i < self.num_leaders else 80
            marker = 's' if i < self.num_leaders else 'o'
            self.ax_zoom.scatter(pos[0], pos[1], color=color, s=size,
                               marker=marker, edgecolors='black',
                               linewidth=2, zorder=15)
        
        # ========== 门控权重图 ==========
        if len(self.gate_weights_history) > 10:
            self.ax_gates.cla()
            self.ax_gates.set_xlabel('Steps', fontsize=13)
            self.ax_gates.set_ylabel('Gate Weight', fontsize=13)
            self.ax_gates.set_title('(c) Dual-Channel Gate Weights', fontsize=14, fontweight='bold')
            self.ax_gates.grid(True, alpha=0.3)
            self.ax_gates.set_ylim([0, 1])
            
            # 计算平均门控权重
            gate_array = np.array(self.gate_weights_history)
            mean_gates = gate_array.mean(axis=1)
            
            steps = np.arange(len(mean_gates))
            self.ax_gates.plot(steps, mean_gates[:, 0], label='Self Channel',
                             linewidth=2, alpha=0.8, color='blue')
            self.ax_gates.plot(steps, mean_gates[:, 1], label='Neighbor Channel',
                             linewidth=2, alpha=0.8, color='green')
            self.ax_gates.plot(steps, mean_gates[:, 2], label='Leader Channel',
                             linewidth=2, alpha=0.8, color='red')
            
            self.ax_gates.legend(fontsize=11, loc='upper right')
        
        # 刷新
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0.001)
    
    def _save_final_visualization(self):
        """保存最终可视化"""
        filename = 'gnn_sac_obstacle_avoidance_result.png'
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 可视化已保存: {filename}")
    
    def _print_summary(self, steps, episode_reward, info, elapsed_time):
        """打印总结"""
        print("\n" + "=" * 80)
        print("实验完成总结")
        print("=" * 80)
        print(f"总步数: {steps}")
        print(f"总时间: {elapsed_time:.2f}秒")
        print(f"总奖励: {episode_reward:.2f}")
        print(f"最终编队误差: {info['formation_error']:.2f}m")
        print(f"最终目标距离: {info['goal_distance']:.2f}m")
        print(f"碰撞次数: {info['collision_count']}")
        
        # 门控权重统计
        if len(self.gate_weights_history) > 0:
            mean_gates = np.array(self.gate_weights_history).mean(axis=(0, 1))
            print(f"\n平均门控权重:")
            print(f"  自身通道: {mean_gates[0]:.3f}")
            print(f"  邻居通道: {mean_gates[1]:.3f}")
            print(f"  领航者通道: {mean_gates[2]:.3f}")
        
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='GNN-SAC障碍物避障编队')
    
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                       help='运行模式')
    parser.add_argument('--num-auv', type=int, default=5, help='AUV数量')
    parser.add_argument('--num-leaders', type=int, default=2, help='领航者数量')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--max-steps', type=int, default=2000, help='最大步数')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    parser.add_argument('--deterministic', action='store_true', help='是否使用确定性策略')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("请使用 train_gnn_sac_formation.py 进行训练")
        return
    
    # 测试模式
    system = GNNSACObstacleAvoidanceFormation(
        num_auv=args.num_auv,
        num_leaders=args.num_leaders,
        env_size=300.0,
        checkpoint_path=args.checkpoint
    )
    
    system.run_experiment(
        max_steps=args.max_steps,
        visualize=args.visualize,
        deterministic=args.deterministic
    )
    
    try:
        input("\n按回车键退出...")
    except Exception:
        pass


if __name__ == "__main__":
    main()