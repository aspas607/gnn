# test_gnn_sac_formation.py
"""
GNN-SAC编队测试与可视化脚本

使用方法:
python test_gnn_sac_formation.py --checkpoint checkpoints_gnn_sac/gnn_sac_final.pth --visualize
"""

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from gnn_sac_formation_env import GNNSACFormationEnv
from gnn_sac_trainer import GNNSACTrainer


class GNNSACVisualizer:
    """GNN-SAC可视化器"""
    
    def __init__(self, env: GNNSACFormationEnv, trainer: GNNSACTrainer):
        self.env = env
        self.trainer = trainer
        
        # 颜色方案
        self.colors = {
            'leader': '#FF0000',
            'follower_1': '#45B7D1',
            'follower_2': '#4ECDC4',
            'follower_3': '#96CEB4'
        }
        
        # 历史记录
        self.positions_history = []
        self.gate_weights_history = []
        
    def run_episode(self, deterministic: bool = True, max_steps: int = 2000):
        """运行一个episode并记录轨迹"""
        
        print("\n" + "=" * 80)
        print("运行测试Episode")
        print("=" * 80)
        
        obs_list = self.env.reset()
        self.positions_history = []
        self.gate_weights_history = []
        
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            step += 1
            
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
            
            # 打印进度
            if step % 100 == 0:
                print(f"Step {step}: "
                      f"Reward={np.sum(rewards):.2f}, "
                      f"Formation Error={info['formation_error']:.2f}m, "
                      f"Goal Dist={info['goal_distance']:.2f}m")
            
            # 检查终止
            if np.all(dones):
                print(f"\n✓ Episode完成于步骤 {step}")
                break
        
        print(f"总奖励: {episode_reward:.2f}")
        print(f"最终编队误差: {info['formation_error']:.2f}m")
        print(f"最终目标距离: {info['goal_distance']:.2f}m")
        print(f"碰撞次数: {info['collision_count']}")
        print("=" * 80 + "\n")
        
        return {
            'episode_reward': episode_reward,
            'steps': step,
            'final_formation_error': info['formation_error'],
            'final_goal_distance': info['goal_distance'],
            'collision_count': info['collision_count']
        }
    
    def visualize_trajectory(self, save_path: str = None):
        """可视化轨迹"""
        
        fig = plt.figure(figsize=(20, 10))
        
        # (a) 整体轨迹图
        ax1 = fig.add_subplot(121)
        ax1.set_xlim(0, self.env.env_size)
        ax1.set_ylim(0, self.env.env_size)
        ax1.set_xlabel('X Location (m)', fontsize=13)
        ax1.set_ylabel('Y Location (m)', fontsize=13)
        ax1.set_title('(a) Trajectory Visualization', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 绘制障碍物
        self.env.obstacle_env.visualize(ax1, show_detection_range=False)
        
        # 绘制起点和终点
        ax1.plot(self.env.start_pos[0], self.env.start_pos[1], 'go',
                markersize=12, label='Start', zorder=20)
        ax1.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'y*',
                markersize=18, label='Goal', zorder=20)
        
        # 绘制轨迹
        positions_array = np.array(self.positions_history)
        for i in range(self.env.num_agents):
            if i < self.env.num_leaders:
                color = self.colors['leader']
                label = f'Leader-{i}'
            else:
                color = self.colors[f'follower_{min(i - self.env.num_leaders, 2) + 1}']
                label = f'Follower-{i - self.env.num_leaders}'
            
            trajectory = positions_array[:, i, :]
            ax1.plot(trajectory[:, 0], trajectory[:, 1],
                    color=color, linewidth=2.5, alpha=0.7, label=label)
            
            # 标记最终位置
            ax1.scatter(trajectory[-1, 0], trajectory[-1, 1],
                       color=color, s=150, marker='o',
                       edgecolors='black', linewidth=2, zorder=15)
        
        ax1.legend(fontsize=10, loc='upper left')
        
        # (b) 门控权重演化
        ax2 = fig.add_subplot(122)
        gate_weights_array = np.array(self.gate_weights_history)
        
        # 平均所有智能体的门控权重
        mean_gate_weights = gate_weights_array.mean(axis=1)
        
        steps = np.arange(len(mean_gate_weights))
        ax2.plot(steps, mean_gate_weights[:, 0], label='Self Channel', 
                linewidth=2, alpha=0.8)
        ax2.plot(steps, mean_gate_weights[:, 1], label='Neighbor Channel',
                linewidth=2, alpha=0.8)
        ax2.plot(steps, mean_gate_weights[:, 2], label='Leader Channel',
                linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Steps', fontsize=13)
        ax2.set_ylabel('Gate Weight', fontsize=13)
        ax2.set_title('(b) Dual-Channel Gate Weights Evolution', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ 轨迹可视化已保存: {save_path}")
        
        plt.show()
    
    def create_animation(self, save_path: str = None, fps: int = 10):
        """创建动画"""
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0, self.env.env_size)
        ax.set_ylim(0, self.env.env_size)
        ax.set_xlabel('X Location (m)', fontsize=12)
        ax.set_ylabel('Y Location (m)', fontsize=12)
        ax.set_title('GNN-SAC Formation Animation', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制静态元素
        self.env.obstacle_env.visualize(ax, show_detection_range=False)
        ax.plot(self.env.start_pos[0], self.env.start_pos[1], 'go', markersize=12)
        ax.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'y*', markersize=18)
        
        # 初始化动态元素
        agent_markers = []
        trail_lines = []
        
        for i in range(self.env.num_agents):
            if i < self.env.num_leaders:
                color = self.colors['leader']
            else:
                color = self.colors[f'follower_{min(i - self.env.num_leaders, 2) + 1}']
            
            marker, = ax.plot([], [], 'o', color=color, markersize=12,
                            markeredgecolor='black', markeredgewidth=2)
            trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
            
            agent_markers.append(marker)
            trail_lines.append(trail)
        
        # 步骤文本
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            for marker, trail in zip(agent_markers, trail_lines):
                marker.set_data([], [])
                trail.set_data([], [])
            step_text.set_text('')
            return agent_markers + trail_lines + [step_text]
        
        def update(frame):
            positions = self.positions_history[frame]
            
            for i, (marker, trail) in enumerate(zip(agent_markers, trail_lines)):
                # 更新位置
                marker.set_data([positions[i, 0]], [positions[i, 1]])
                
                # 更新轨迹
                if frame > 0:
                    trail_data = np.array(self.positions_history[:frame+1])[:, i, :]
                    trail.set_data(trail_data[:, 0], trail_data[:, 1])
            
            step_text.set_text(f'Step: {frame}/{len(self.positions_history)-1}')
            
            return agent_markers + trail_lines + [step_text]
        
        # 创建动画
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=len(self.positions_history),
                           interval=100, blit=True)
        
        if save_path:
            print("正在保存动画（这可能需要几分钟）...")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"✓ 动画已保存: {save_path}")
        
        plt.close()
        return anim


def test_trained_model(args):
    """测试训练好的模型"""
    
    print("=" * 80)
    print("GNN-SAC编队测试")
    print("=" * 80)
    print(f"检查点: {args.checkpoint}")
    print(f"设备: {args.device}")
    print("=" * 80 + "\n")
    
    # 创建环境
    env = GNNSACFormationEnv(
        num_agents=args.num_agents,
        num_leaders=args.num_leaders,
        env_size=300.0,
        num_obstacles=5
    )
    
    # 创建训练器
    trainer = GNNSACTrainer(
        num_agents=args.num_agents,
        num_leaders=args.num_leaders,
        device=args.device
    )
    
    # 加载检查点
    trainer.load_checkpoint(args.checkpoint)
    trainer.actor.eval()
    
    # 创建可视化器
    visualizer = GNNSACVisualizer(env, trainer)
    
    # 运行episode
    results = visualizer.run_episode(deterministic=True, max_steps=2000)
    
    # 可视化
    if args.visualize:
        visualizer.visualize_trajectory(
            save_path='gnn_sac_trajectory.png' if args.save_fig else None
        )
    
    # 创建动画
    if args.animate:
        visualizer.create_animation(
            save_path='gnn_sac_animation.gif' if args.save_fig else None,
            fps=10
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试GNN-SAC编队')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    parser.add_argument('--num-agents', type=int, default=5, help='智能体数量')
    parser.add_argument('--num-leaders', type=int, default=2, help='领航者数量')
    parser.add_argument('--visualize', action='store_true', help='是否可视化')
    parser.add_argument('--animate', action='store_true', help='是否创建动画')
    parser.add_argument('--save-fig', action='store_true', help='是否保存图像')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 测试模型
    results = test_trained_model(args)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print(f"总奖励: {results['episode_reward']:.2f}")
    print(f"步数: {results['steps']}")
    print(f"最终编队误差: {results['final_formation_error']:.2f}m")
    print(f"最终目标距离: {results['final_goal_distance']:.2f}m")
    print(f"碰撞次数: {results['collision_count']}")
    print("=" * 80)


if __name__ == "__main__":
    main()