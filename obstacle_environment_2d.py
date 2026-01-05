# obstacle_environment_2d.py
"""
二维障碍物环境 - 用于AUV编队避障测试
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Dict


class Obstacle:
    """圆形障碍物类"""
    
    def __init__(self, center: np.ndarray, radius: float):
        """
        初始化障碍物
        
        Args:
            center: 障碍物中心 [x, y]
            radius: 障碍物半径
        """
        self.center = center
        self.radius = radius
        self.safety_margin = 2.0  # 安全边距
        
    def check_collision(self, position: np.ndarray, agent_radius: float = 1.0) -> bool:
        """
        检查是否与障碍物碰撞
        
        Args:
            position: 检查位置 [x, y]
            agent_radius: 智能体半径
            
        Returns:
            是否碰撞
        """
        distance = np.linalg.norm(position - self.center)
        return distance < (self.radius + agent_radius + self.safety_margin)
    
    def get_repulsive_force(self, position: np.ndarray,
                           detection_range: float = 30.0,
                           max_force: float = 80.0) -> np.ndarray:
        """
        计算斥力（用于避障）- 优化版：使用线性衰减而非平方反比

        Args:
            position: 当前位置
            detection_range: 检测范围
            max_force: 最大斥力

        Returns:
            斥力向量
        """
        diff = position - self.center
        distance = np.linalg.norm(diff)

        # 如果在检测范围外，不产生斥力
        if distance > detection_range + self.radius:
            return np.zeros(2)

        # 计算斥力大小（改用线性衰减，避免距离近时斥力过大）
        effective_distance = distance - self.radius
        if effective_distance < 0.1:
            effective_distance = 0.1

        # 线性衰减公式：距离越近，斥力越大，但增长更平缓
        # force_magnitude = max_force * (1 - effective_distance / detection_range)
        # 使用指数为1.2的幂次，介于线性和平方之间
        force_magnitude = max_force * (detection_range / effective_distance) ** 1.2
        force_magnitude = min(force_magnitude, max_force)

        # 斥力方向：远离障碍物
        if distance > 0.01:
            force_direction = diff / distance
        else:
            force_direction = np.random.randn(2)
            force_direction = force_direction / np.linalg.norm(force_direction)

        return force_direction * force_magnitude


class ObstacleEnvironment2D:
    """二维障碍物环境"""
    
    def __init__(self, 
                 env_size: float = 300.0,
                 num_obstacles: int = 5,
                 seed: int = 42):
        """
        初始化二维障碍环境
        
        Args:
            env_size: 环境尺寸
            num_obstacles: 障碍物数量
            seed: 随机种子
        """
        self.env_size = env_size
        self.num_obstacles = num_obstacles
        
        np.random.seed(seed)
        
        # 生成障碍物
        self.obstacles = self._generate_obstacles()
        
        print(f"✓ 二维障碍环境初始化完成")
        print(f"  环境尺寸: {env_size}m × {env_size}m")
        print(f"  障碍物数量: {len(self.obstacles)}")
        
    def _generate_obstacles(self) -> List[Obstacle]:
        """生成障碍物分布（精确复刻图2 - 5个障碍物）"""
        obstacles = []
        
        # 修改后的5个障碍物位置和大小
        # 从左下到右上排列
        obstacle_configs = [
            # (x, y, radius) - 更新为用户指定的配置
            (55, 75, 30),      # 第1个：左下障碍 (更新为 x=55)
            (140, 75, 35),     # 第2个：中下大障碍 (保持不变)
            (150, 150, 15),    # 第3个：中间障碍 (从155->150)
            (220, 150, 30),    # 第4个：右上大障碍 (保持不变)
            (220, 235, 25),    # 第5个：最右上障碍 (从235->220)
        ]
        
        # 根据num_obstacles参数选择障碍物数量
        actual_num = min(self.num_obstacles, len(obstacle_configs))
        
        for i, (x, y, radius) in enumerate(obstacle_configs[:actual_num]):
            center = np.array([x, y])
            obstacle = Obstacle(center, radius)
            obstacles.append(obstacle)
            print(f"  障碍物{i+1}: 中心({x:.0f}, {y:.0f}), 半径{radius:.0f}m")
        
        return obstacles
    
    def check_collision(self, position: np.ndarray, agent_radius: float = 1.0) -> bool:
        """
        检查位置是否与任何障碍物碰撞
        
        Args:
            position: 检查位置
            agent_radius: 智能体半径
            
        Returns:
            是否碰撞
        """
        for obstacle in self.obstacles:
            if obstacle.check_collision(position, agent_radius):
                return True
        return False
    
    def get_total_repulsive_force(self, position: np.ndarray,
                                  detection_range: float = 30.0) -> np.ndarray:
        """
        计算所有障碍物的总斥力
        
        Args:
            position: 当前位置
            detection_range: 检测范围
            
        Returns:
            总斥力向量
        """
        total_force = np.zeros(2)
        
        for obstacle in self.obstacles:
            force = obstacle.get_repulsive_force(position, detection_range)
            total_force += force
        
        return total_force
    
    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """
        获取到最近障碍物的距离
        
        Args:
            position: 当前位置
            
        Returns:
            最近距离
        """
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle.center) - obstacle.radius
            if distance < min_distance:
                min_distance = distance
        
        return max(0, min_distance)
    
    def visualize(self, ax, show_detection_range: bool = False):
        """
        可视化障碍物
        
        Args:
            ax: matplotlib坐标轴
            show_detection_range: 是否显示检测范围
        """
        for i, obstacle in enumerate(self.obstacles):
            # 绘制障碍物
            circle = Circle(obstacle.center, obstacle.radius,
                          color='cornflowerblue', alpha=0.7,
                          edgecolor='darkblue', linewidth=2,
                          zorder=5)
            ax.add_patch(circle)
            
            # 可选：显示检测范围
            if show_detection_range:
                detection_circle = Circle(obstacle.center, 
                                        obstacle.radius + 15.0,
                                        color='lightblue', alpha=0.2,
                                        linestyle='--', fill=False,
                                        zorder=3)
                ax.add_patch(detection_circle)
    
    def is_path_clear(self, start: np.ndarray, end: np.ndarray,
                     num_samples: int = 20) -> bool:
        """
        检查两点之间的直线路径是否无碰撞
        
        Args:
            start: 起点
            end: 终点
            num_samples: 采样点数量
            
        Returns:
            路径是否清晰
        """
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start + t * (end - start)
            
            if self.check_collision(point):
                return False
        
        return True
    
    def get_environment_info(self) -> Dict:
        """获取环境信息"""
        return {
            'env_size': self.env_size,
            'num_obstacles': len(self.obstacles),
            'obstacles': [
                {
                    'center': obs.center.tolist(),
                    'radius': obs.radius
                }
                for obs in self.obstacles
            ]
        }


def test_environment():
    """测试环境"""
    print("=" * 60)
    print("二维障碍环境测试")
    print("=" * 60)
    
    # 创建环境
    env = ObstacleEnvironment2D(
        env_size=300.0,
        num_obstacles=5,
        seed=42
    )
    
    # 测试碰撞检测
    test_positions = [
        np.array([10, 10]),
        np.array([60, 50]),
        np.array([150, 150]),
    ]
    
    print("\n碰撞检测测试:")
    for pos in test_positions:
        collision = env.check_collision(pos)
        distance = env.get_nearest_obstacle_distance(pos)
        print(f"  位置 {pos}: 碰撞={collision}, 最近障碍物距离={distance:.1f}m")
    
    # 可视化
    print("\n生成可视化...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置坐标轴
    ax.set_xlim(0, env.env_size)
    ax.set_ylim(0, env.env_size)
    ax.set_xlabel('X Location (m)', fontsize=12)
    ax.set_ylabel('Y Location (m)', fontsize=12)
    ax.set_title('2D Obstacle Environment (5 Obstacles)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制障碍物
    env.visualize(ax, show_detection_range=True)
    
    # 标记起点和终点
    start_pos = np.array([15, 15])
    goal_pos = np.array([285, 285])
    
    ax.plot(start_pos[0], start_pos[1], 'ro', markersize=12,
            label='Start', zorder=10)
    ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=20,
            label='Goal', zorder=10)
    
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('obstacle_environment_2d_5obs.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化已保存: obstacle_environment_2d_5obs.png")
    plt.show()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_environment()