# train_gnn_sac_formation.py
"""
GNN-SAC编队训练脚本

使用方法:
python train_gnn_sac_formation.py --episodes 5000 --save-interval 100
"""

import numpy as np
import torch
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from gnn_sac_formation_env import GNNSACFormationEnv
from gnn_sac_trainer import GNNSACTrainer


def train_gnn_sac(args):
    """训练GNN-SAC编队"""
    
    print("=" * 80)
    print("GNN-SAC双通道聚合编队训练")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"训练episodes: {args.episodes}")
    print("=" * 80 + "\n")
    
    # 创建环境
    env = GNNSACFormationEnv(
        num_agents=args.num_agents,
        num_leaders=args.num_leaders,
        env_size=300.0,
        num_obstacles=5,
        max_neighbors=args.max_neighbors
    )
    
    # 创建训练器
    trainer = GNNSACTrainer(
        num_agents=args.num_agents,
        num_leaders=args.num_leaders,
        state_dim=8,
        neighbor_dim=6,
        leader_dim=5,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        auto_alpha=True,
        device=args.device
    )
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    collision_counts = []
    formation_errors = []
    goal_distances = []
    gate_weight_logs = []
    
    # 训练循环
    for episode in range(args.episodes):
        # 重置环境
        obs_list = env.reset()
        episode_reward = 0
        step = 0
        episode_gate_weights = []
        
        while True:
            step += 1
            
            # 选择动作（前warmup_steps步使用随机策略）
            if trainer.train_step < args.warmup_steps:
                actions = np.random.uniform(-1, 1, (args.num_agents, 2))
                gate_weights = None
            else:
                actions = []
                gate_weights_batch = []
                for obs in obs_list:
                    action, gate_w = trainer.select_action(obs, deterministic=False)
                    actions.append(action[0])
                    gate_weights_batch.append(gate_w[0])
                actions = np.array(actions)
                gate_weights = np.array(gate_weights_batch)
                
                # 记录门控权重
                if gate_weights is not None:
                    episode_gate_weights.append(gate_weights.mean(axis=0))
            
            # 环境步进
            next_obs_list, rewards, dones, info = env.step(actions)
            
            # 存储经验
            for i in range(args.num_agents):
                trainer.replay_buffer.add(
                    state=obs_list[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_obs_list[i],
                    done=dones[i]
                )
            
            # 训练
            if trainer.train_step >= args.warmup_steps and len(trainer.replay_buffer) > args.batch_size:
                for _ in range(args.train_steps_per_env_step):
                    train_info = trainer.train_step_sac(batch_size=args.batch_size)
            
            # 更新状态
            obs_list = next_obs_list
            episode_reward += np.sum(rewards)
            
            # 检查终止
            if np.all(dones):
                break
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        collision_counts.append(info['collision_count'])
        formation_errors.append(info['formation_error'])
        goal_distances.append(info['goal_distance'])
        
        if len(episode_gate_weights) > 0:
            gate_weight_logs.append(np.mean(episode_gate_weights, axis=0))
        
        # 打印进度
        if (episode + 1) % args.print_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.print_interval:])
            avg_length = np.mean(episode_lengths[-args.print_interval:])
            avg_collision = np.mean(collision_counts[-args.print_interval:])
            avg_formation_error = np.mean(formation_errors[-args.print_interval:])
            avg_goal_dist = np.mean(goal_distances[-args.print_interval:])
            
            print(f"Episode {episode+1}/{args.episodes}:")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均步数: {avg_length:.0f}")
            print(f"  平均碰撞: {avg_collision:.2f}")
            print(f"  编队误差: {avg_formation_error:.2f}m")
            print(f"  目标距离: {avg_goal_dist:.2f}m")
            
            if trainer.auto_alpha and len(trainer.alpha_values) > 0:
                print(f"  温度α: {trainer.alpha_values[-1]:.4f}")
            
            if len(gate_weight_logs) > 0:
                mean_gate = gate_weight_logs[-1]
                print(f"  门控权重: [自身={mean_gate[0]:.2f}, 邻居={mean_gate[1]:.2f}, 领航={mean_gate[2]:.2f}]")
            print()
        
        # 保存检查点
        if (episode + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"gnn_sac_ep{episode+1}.pth")
            trainer.save_checkpoint(save_path)
            
            # 保存训练曲线
            plot_training_curves(
                episode_rewards, episode_lengths, collision_counts,
                formation_errors, goal_distances, gate_weight_logs,
                save_path=os.path.join(args.save_dir, f"training_curves_ep{episode+1}.png")
            )
    
    # 最终保存
    final_path = os.path.join(args.save_dir, "gnn_sac_final.pth")
    trainer.save_checkpoint(final_path)
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最终平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"模型保存于: {final_path}")
    print("=" * 80)
    
    return trainer, env


def plot_training_curves(episode_rewards, episode_lengths, collision_counts,
                         formation_errors, goal_distances, gate_weight_logs,
                         save_path: str = None):
    """绘制训练曲线"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GNN-SAC训练曲线', fontsize=16, fontweight='bold')
    
    # 1. Episode奖励
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color='blue')
    # 移动平均
    if len(episode_rewards) > 50:
        window = 50
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label='MA-50')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Episode长度
    ax = axes[0, 1]
    ax.plot(episode_lengths, alpha=0.5, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    
    # 3. 碰撞次数
    ax = axes[0, 2]
    ax.plot(collision_counts, alpha=0.5, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collisions')
    ax.set_title('Collision Count')
    ax.grid(True, alpha=0.3)
    
    # 4. 编队误差
    ax = axes[1, 0]
    ax.plot(formation_errors, alpha=0.5, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error (m)')
    ax.set_title('Formation Error')
    ax.grid(True, alpha=0.3)
    
    # 5. 目标距离
    ax = axes[1, 1]
    ax.plot(goal_distances, alpha=0.5, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Goal Distance')
    ax.grid(True, alpha=0.3)
    
    # 6. 门控权重
    ax = axes[1, 2]
    if len(gate_weight_logs) > 0:
        gate_array = np.array(gate_weight_logs)
        ax.plot(gate_array[:, 0], label='Self', alpha=0.7)
        ax.plot(gate_array[:, 1], label='Neighbor', alpha=0.7)
        ax.plot(gate_array[:, 2], label='Leader', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Weight')
        ax.set_title('Gate Weights (Mean)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练GNN-SAC编队')
    
    # 环境参数
    parser.add_argument('--num-agents', type=int, default=5, help='智能体数量')
    parser.add_argument('--num-leaders', type=int, default=2, help='领航者数量')
    parser.add_argument('--max-neighbors', type=int, default=4, help='最大邻居数')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=3000, help='训练episodes')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='预热步数')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--train-steps-per-env-step', type=int, default=1, help='每个环境步的训练步数')
    
    # 网络参数
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='Actor学习率')
    parser.add_argument('--lr-critic', type=float, default=3e-4, help='Critic学习率')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_gnn_sac', help='保存目录')
    parser.add_argument('--print-interval', type=int, default=10, help='打印间隔')
    parser.add_argument('--save-interval', type=int, default=100, help='保存间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 开始训练
    trainer, env = train_gnn_sac(args)


if __name__ == "__main__":
    main()