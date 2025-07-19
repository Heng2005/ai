"""
MNIST环境适配器 - 将MNIST数据集转换为强化学习环境
"""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
import random
from config import TRAINING_CONFIG


class MNISTClassificationEnv(gym.Env):
    """
    MNIST分类环境 - 将手写数字识别转换为强化学习问题
    """
    
    def __init__(self, x_data: torch.Tensor, y_data: torch.Tensor, max_steps: int = 1000):
        super(MNISTClassificationEnv, self).__init__()
        
        self.x_data = x_data
        self.y_data = y_data
        self.max_steps = max_steps
        self.current_step = 0
        self.current_index = 0
        
        # 动作空间：10个数字类别 (0-9)
        self.action_space = spaces.Discrete(10)
        
        # 观察空间：28x28的灰度图像，展平为784维向量
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(784,), dtype=np.float32
        )
        
        # 性能统计
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        self.current_index = random.randint(0, len(self.x_data) - 1)
        
        # 获取当前图像并展平
        current_image = self.x_data[self.current_index].numpy().flatten()
        
        info = {
            'true_label': int(self.y_data[self.current_index].item()),
            'image_index': self.current_index
        }
        
        return current_image.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        self.current_step += 1
        self.total_predictions += 1
        
        # 获取真实标签
        true_label = int(self.y_data[self.current_index].item())
        
        # 计算奖励
        if action == true_label:
            reward = 1.0  # 正确预测
            self.correct_predictions += 1
        else:
            reward = -0.1  # 错误预测
        
        # 检查是否结束
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # 准备下一个观察
        if not (terminated or truncated):
            self.current_index = random.randint(0, len(self.x_data) - 1)
        
        next_obs = self.x_data[self.current_index].numpy().flatten().astype(np.float32)
        
        info = {
            'true_label': true_label,
            'predicted_label': action,
            'correct': action == true_label,
            'accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0,
            'image_index': self.current_index
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


class MNISTSequentialEnv(gym.Env):
    """
    MNIST顺序环境 - 按顺序处理MNIST数据集
    """
    
    def __init__(self, x_data: torch.Tensor, y_data: torch.Tensor):
        super(MNISTSequentialEnv, self).__init__()
        
        self.x_data = x_data
        self.y_data = y_data
        self.current_index = 0
        self.total_samples = len(x_data)
        
        # 动作空间：10个数字类别 (0-9)
        self.action_space = spaces.Discrete(10)
        
        # 观察空间：28x28的灰度图像，展平为784维向量
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(784,), dtype=np.float32
        )
        
        # 性能统计
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_index = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # 获取第一个图像
        current_image = self.x_data[self.current_index].numpy().flatten()
        
        info = {
            'true_label': int(self.y_data[self.current_index].item()),
            'image_index': self.current_index,
            'progress': self.current_index / self.total_samples
        }
        
        return current_image.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        self.total_predictions += 1
        
        # 获取真实标签
        true_label = int(self.y_data[self.current_index].item())
        
        # 计算奖励
        if action == true_label:
            reward = 1.0
            self.correct_predictions += 1
        else:
            reward = -0.1
        
        # 移动到下一个样本
        self.current_index += 1
        
        # 检查是否完成所有样本
        terminated = self.current_index >= self.total_samples
        truncated = False
        
        # 准备下一个观察
        if not terminated:
            next_obs = self.x_data[self.current_index].numpy().flatten().astype(np.float32)
        else:
            # 如果结束，返回零向量
            next_obs = np.zeros(784, dtype=np.float32)
        
        info = {
            'true_label': true_label,
            'predicted_label': action,
            'correct': action == true_label,
            'accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0,
            'image_index': self.current_index - 1,
            'progress': (self.current_index - 1) / self.total_samples,
            'total_correct': self.correct_predictions,
            'total_samples': self.total_predictions
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        pass
    
    def close(self):
        """关闭环境"""
        pass


def create_mnist_env(x_data: torch.Tensor, y_data: torch.Tensor, env_type: str = 'random') -> gym.Env:
    """
    创建MNIST环境
    
    Args:
        x_data: 图像数据
        y_data: 标签数据
        env_type: 环境类型 ('random' 或 'sequential')
    
    Returns:
        gym.Env: MNIST环境实例
    """
    if env_type == 'random':
        return MNISTClassificationEnv(x_data, y_data)
    elif env_type == 'sequential':
        return MNISTSequentialEnv(x_data, y_data)
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
