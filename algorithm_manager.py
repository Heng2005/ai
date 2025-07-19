"""
算法管理器 - 统一管理不同的机器学习算法
"""
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mnist_env import create_mnist_env
from config import SB3_ALGORITHMS, MODEL_PATH, TRAINING_CONFIG
import wandb


class TrainingCallback(BaseCallback):
    """自定义训练回调，用于记录训练过程"""
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 记录每步的信息
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        return True


class BaseAlgorithm(ABC):
    """算法基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None
        self.training_time = 0
        self.inference_time = 0
        self.model_size = 0
        
    @abstractmethod
    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, 
              x_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, Any]:
        """训练算法"""
        pass
    
    @abstractmethod
    def predict(self, x_test: torch.Tensor) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """评估算法性能"""
        pass
    
    def save_model(self, path: str):
        """保存模型"""
        pass
    
    def load_model(self, path: str):
        """加载模型"""
        pass


class PyTorchCNN(BaseAlgorithm):
    """PyTorch CNN算法"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PyTorch_CNN", config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()
        
    def _build_model(self):
        """构建CNN模型"""
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
                self.pool1 = nn.MaxPool2d(kernel_size=2)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
                self.pool2 = nn.MaxPool2d(kernel_size=2)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
                self.pool3 = nn.MaxPool2d(kernel_size=2)
                self.fc1 = nn.Linear(128, 128)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool1(x)
                x = torch.relu(self.conv2(x))
                x = self.pool2(x)
                x = torch.relu(self.conv3(x))
                x = self.pool3(x)
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        self.model = CNN().to(self.device)
        
    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, 
              x_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, Any]:
        """训练CNN模型"""
        start_time = time.time()
        
        # 创建数据加载器
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            # 记录历史
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(MODEL_PATH, f'{self.name}_best.pth'))
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]} - '
                  f'Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%')
        
        self.training_time = time.time() - start_time
        return history
    
    def predict(self, x_test: torch.Tensor) -> np.ndarray:
        """预测"""
        start_time = time.time()
        
        self.model.eval()
        predictions = []
        
        test_loader = DataLoader(TensorDataset(x_test), batch_size=1000)
        
        with torch.no_grad():
            for batch_x, in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        self.inference_time = time.time() - start_time
        return np.array(predictions)
    
    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """评估模型"""
        predictions = self.predict(x_test)
        y_true = y_test.numpy()
        
        accuracy = np.mean(predictions == y_true) * 100
        
        # 计算模型大小
        self.model_size = sum(p.numel() for p in self.model.parameters()) * 4 / (1024 * 1024)  # MB
        
        return {
            'accuracy': accuracy,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size_mb': self.model_size
        }


class StableBaselinesAlgorithm(BaseAlgorithm):
    """Stable Baselines3算法包装器"""
    
    def __init__(self, algorithm_name: str, config: Dict[str, Any]):
        super().__init__(f"SB3_{algorithm_name}", config)
        self.algorithm_name = algorithm_name
        self.algorithm_class = self._get_algorithm_class()
        
    def _get_algorithm_class(self):
        """获取算法类"""
        algorithms = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN,
            'SAC': SAC
        }
        return algorithms[self.algorithm_name]
    
    def train(self, x_train: torch.Tensor, y_train: torch.Tensor, 
              x_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, Any]:
        """训练强化学习算法"""
        start_time = time.time()
        
        # 创建环境
        env = create_mnist_env(x_train, y_train, env_type='random')
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # 创建模型
        model_config = {k: v for k, v in self.config.items() if k != 'total_timesteps'}
        self.model = self.algorithm_class(env=env, **model_config)
        
        # 训练回调
        callback = TrainingCallback()
        
        # 训练
        self.model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callback
        )
        
        self.training_time = time.time() - start_time
        
        # 保存模型
        self.model.save(os.path.join(MODEL_PATH, f'{self.name}_model'))
        
        return {
            'episode_rewards': callback.episode_rewards,
            'episode_lengths': callback.episode_lengths,
            'training_time': self.training_time
        }
    
    def predict(self, x_test: torch.Tensor) -> np.ndarray:
        """预测"""
        start_time = time.time()
        
        # 创建测试环境
        dummy_labels = torch.zeros(len(x_test), dtype=torch.long)
        test_env = create_mnist_env(x_test, dummy_labels, env_type='sequential')
        
        predictions = []
        obs, _ = test_env.reset()
        
        for i in range(len(x_test)):
            action, _ = self.model.predict(obs, deterministic=True)
            predictions.append(action)
            
            if i < len(x_test) - 1:
                obs, _, terminated, truncated, _ = test_env.step(action)
                if terminated or truncated:
                    break
        
        self.inference_time = time.time() - start_time
        return np.array(predictions)
    
    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """评估模型"""
        predictions = self.predict(x_test)
        y_true = y_test.numpy()
        
        accuracy = np.mean(predictions == y_true) * 100
        
        return {
            'accuracy': accuracy,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size_mb': 0  # SB3模型大小计算较复杂，暂时设为0
        }


class AlgorithmManager:
    """算法管理器"""
    
    def __init__(self):
        self.algorithms = {}
        self.results = {}
        
    def register_algorithm(self, algorithm: BaseAlgorithm):
        """注册算法"""
        self.algorithms[algorithm.name] = algorithm
        
    def train_all(self, x_train: torch.Tensor, y_train: torch.Tensor, 
                  x_val: torch.Tensor, y_val: torch.Tensor):
        """训练所有算法"""
        for name, algorithm in self.algorithms.items():
            print(f"\n开始训练 {name}...")
            
            # 开始WandB运行
            run = wandb.init(
                project="handwritten-recognition-comparison",
                name=f"{name}_experiment",
                tags=[name.lower(), "mnist"],
                reinit=True
            )
            
            try:
                history = algorithm.train(x_train, y_train, x_val, y_val)
                self.results[name] = {'training_history': history}
                
                # 记录训练历史到WandB
                if 'train_acc' in history:
                    for epoch, (train_acc, val_acc) in enumerate(zip(history['train_acc'], history['val_acc'])):
                        wandb.log({
                            'epoch': epoch,
                            'train_accuracy': train_acc,
                            'val_accuracy': val_acc
                        })
                
                print(f"{name} 训练完成")
                
            except Exception as e:
                print(f"{name} 训练失败: {e}")
                self.results[name] = {'error': str(e)}
            
            finally:
                wandb.finish()
    
    def evaluate_all(self, x_test: torch.Tensor, y_test: torch.Tensor):
        """评估所有算法"""
        for name, algorithm in self.algorithms.items():
            if name in self.results and 'error' not in self.results[name]:
                print(f"\n评估 {name}...")
                
                try:
                    metrics = algorithm.evaluate(x_test, y_test)
                    self.results[name]['metrics'] = metrics
                    
                    print(f"{name} 准确率: {metrics['accuracy']:.2f}%")
                    
                except Exception as e:
                    print(f"{name} 评估失败: {e}")
                    self.results[name]['evaluation_error'] = str(e)
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """获取对比结果"""
        return self.results
