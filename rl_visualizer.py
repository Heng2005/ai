"""
强化学习可视化模块 - 展示多个Gymnasium环境下不同强化学习算法的效果
"""
import os
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from tqdm import tqdm

# 设置样式
sns.set_theme(style="darkgrid")


class RLVisualizer:
    """强化学习可视化器"""
    
    def __init__(self, base_dir: str = "results/rl_visualization"):
        """
        初始化可视化器
        
        Args:
            base_dir: 结果保存的基础目录
        """
        self.base_dir = base_dir
        self.video_dir = os.path.join(base_dir, "videos")
        self.model_dir = os.path.join(base_dir, "models")
        self.log_dir = os.path.join(base_dir, "logs")
        
        # 创建必要的目录
        for directory in [self.base_dir, self.video_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 支持的环境列表
        self.supported_envs = {
            "CartPole-v1": {
                "description": "平衡杆问题 - 控制小车平衡一个倒立的杆",
                "type": "classic_control",
                "max_steps": 500,
                "algorithms": ["PPO", "A2C", "DQN"]
            },
            "MountainCar-v0": {
                "description": "爬山小车问题 - 小车需要爬上陡峭的山坡",
                "type": "classic_control",
                "max_steps": 200,
                "algorithms": ["PPO", "A2C", "DQN"]
            },
            "Acrobot-v1": {
                "description": "双节摆问题 - 控制双节摆达到目标高度",
                "type": "classic_control",
                "max_steps": 500,
                "algorithms": ["PPO", "A2C", "DQN"]
            },
            "LunarLander-v3": {
                "description": "月球着陆器 - 控制着陆器安全着陆",
                "type": "box2d",
                "max_steps": 1000,
                "algorithms": ["PPO", "A2C", "DQN"]
            },
            "BipedalWalker-v3": {
                "description": "双足行走机器人 - 控制机器人平稳行走",
                "type": "box2d",
                "max_steps": 1600,
                "algorithms": ["PPO", "A2C"]
            }
        }
        
        # 算法配置
        self.algorithm_configs = {
            "PPO": {
                "class": PPO,
                "params": {
                    "policy": "MlpPolicy",
                    "verbose": 0,
                    "learning_rate": 0.0003,
                }
            },
            "A2C": {
                "class": A2C,
                "params": {
                    "policy": "MlpPolicy",
                    "verbose": 0,
                    "learning_rate": 0.0007,
                }
            },
            "DQN": {
                "class": DQN,
                "params": {
                    "policy": "MlpPolicy",
                    "verbose": 0,
                    "learning_rate": 0.0001,
                    "buffer_size": 50000,
                    "learning_starts": 1000,
                }
            }
        }
        
        # 训练结果
        self.training_results = {}
    
    def list_available_environments(self) -> None:
        """列出所有可用的环境"""
        print("\n可用的强化学习环境:")
        print("=" * 60)
        for i, (env_id, info) in enumerate(self.supported_envs.items(), 1):
            print(f"{i}. {env_id} - {info['description']}")
            print(f"   支持的算法: {', '.join(info['algorithms'])}")
        print("=" * 60)
    
    def create_env(self, env_id: str, seed: int = 42, render_mode: str = None) -> gym.Env:
        """
        创建环境
        
        Args:
            env_id: 环境ID
            seed: 随机种子
            render_mode: 渲染模式，可以是'human'、'rgb_array'或None
            
        Returns:
            gym.Env: Gymnasium环境实例
        """
        if env_id not in self.supported_envs:
            raise ValueError(f"不支持的环境: {env_id}")
        
        # 创建环境
        env = gym.make(env_id, render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        
        return env
    
    def create_vec_env(self, env_id: str, n_envs: int = 1, seed: int = 42, render_mode: str = None) -> DummyVecEnv:
        """
        创建向量化环境
        
        Args:
            env_id: 环境ID
            n_envs: 环境数量
            seed: 随机种子
            render_mode: 渲染模式，可以是'human'、'rgb_array'或None
            
        Returns:
            DummyVecEnv: 向量化环境
        """
        def make_env(env_id, rank, seed, render_mode):
            def _init():
                env = gym.make(env_id, render_mode=render_mode)
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            return _init
        
        env = DummyVecEnv([make_env(env_id, i, seed, render_mode) for i in range(n_envs)])
        return env
    
    def create_recording_env(self, env_id: str, video_folder: str, 
                           video_length: int = 1000, name_prefix: str = "") -> VecVideoRecorder:
        """
        创建录制视频的环境
        
        Args:
            env_id: 环境ID
            video_folder: 视频保存文件夹
            video_length: 视频长度
            name_prefix: 视频名称前缀
            
        Returns:
            VecVideoRecorder: 录制环境
        """
        # 录制视频必须使用rgb_array渲染模式
        env = self.create_vec_env(env_id, render_mode='rgb_array')
        
        # 创建录制环境
        env = VecVideoRecorder(
            env, 
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,  # 只在第一个episode录制
            video_length=video_length,
            name_prefix=name_prefix
        )
        
        return env
    
    def train_model(self, env_id: str, algorithm: str, total_timesteps: int = 100000) -> Tuple[Any, Dict]:
        """
        训练模型
        
        Args:
            env_id: 环境ID
            algorithm: 算法名称
            total_timesteps: 总训练步数
            
        Returns:
            Tuple[Any, Dict]: 训练好的模型和训练结果
        """
        if env_id not in self.supported_envs:
            raise ValueError(f"不支持的环境: {env_id}")
            
        if algorithm not in self.supported_envs[env_id]["algorithms"]:
            raise ValueError(f"环境 {env_id} 不支持算法 {algorithm}")
        
        # 获取算法配置
        algo_config = self.algorithm_configs[algorithm]
        
        # 创建环境 - 训练时不需要渲染
        env = self.create_vec_env(env_id, render_mode=None)
        
        # 创建模型
        model_class = algo_config["class"]
        model_params = algo_config["params"].copy()
        
        # 检查是否安装了tensorboard
        try:
            import tensorboard
            # 添加tensorboard日志
            log_path = os.path.join(self.log_dir, f"{env_id}_{algorithm}")
            model_params["tensorboard_log"] = log_path
        except ImportError:
            # 如果未安装tensorboard，则不使用
            print("注意: Tensorboard未安装，将不使用tensorboard记录训练过程")
        
        print(f"\n开始训练 {env_id} 环境，使用 {algorithm} 算法...")
        start_time = time.time()
        
        # 创建并训练模型
        model = model_class(env=env, **model_params)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{env_id}_{algorithm}")
        model.save(model_path)
        
        # 评估模型
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        # 记录训练结果
        training_time = time.time() - start_time
        results = {
            "env_id": env_id,
            "algorithm": algorithm,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "training_time": training_time,
            "total_timesteps": total_timesteps,
            "model_path": model_path
        }
        
        # 存储结果
        if env_id not in self.training_results:
            self.training_results[env_id] = {}
        self.training_results[env_id][algorithm] = results
        
        print(f"训练完成! 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"训练时间: {training_time:.2f}秒")
        
        return model, results
    
    def record_video(self, env_id: str, algorithm: str, video_length: int = 1000) -> str:
        """
        记录智能体表现的视频
        
        Args:
            env_id: 环境ID
            algorithm: 算法名称
            video_length: 视频长度
            
        Returns:
            str: 视频保存路径
        """
        # 检查模型是否已训练
        if (env_id not in self.training_results or 
            algorithm not in self.training_results[env_id]):
            raise ValueError(f"模型 {env_id}_{algorithm} 尚未训练")
        
        # 加载模型
        model_path = self.training_results[env_id][algorithm]["model_path"]
        algo_class = self.algorithm_configs[algorithm]["class"]
        model = algo_class.load(model_path)
        
        # 创建录制环境
        video_folder = os.path.join(self.video_dir, f"{env_id}_{algorithm}")
        os.makedirs(video_folder, exist_ok=True)
        
        env = self.create_recording_env(
            env_id, 
            video_folder=video_folder,
            video_length=video_length,
            name_prefix=f"{env_id}_{algorithm}"
        )
        
        # 运行模型
        obs = env.reset()
        for _ in range(video_length):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
        
        env.close()
        
        # 查找生成的视频文件
        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if video_files:
            video_path = os.path.join(video_folder, video_files[0])
            print(f"视频已保存至: {video_path}")
            return video_path
        else:
            print("未找到生成的视频文件")
            return ""
    
    def visualize_training_results(self) -> None:
        """可视化训练结果"""
        if not self.training_results:
            print("没有训练结果可供可视化")
            return
        
        # 准备数据
        data = []
        for env_id, env_results in self.training_results.items():
            for algo, results in env_results.items():
                data.append({
                    "环境": env_id,
                    "算法": algo,
                    "平均奖励": results["mean_reward"],
                    "标准差": results["std_reward"],
                    "训练时间(秒)": results["training_time"]
                })
        
        df = pd.DataFrame(data)
        
        # 绘制奖励对比图
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="环境", y="平均奖励", hue="算法", data=df)
        
        # 添加误差条 - 使用列名而不是位置索引
        for i, row in enumerate(df.itertuples()):
            # 获取当前行的环境和算法
            env_name = row.Index if hasattr(row, 'Index') else i
            # 使用DataFrame直接获取值
            mean_reward = df.iloc[i]['平均奖励']
            std_reward = df.iloc[i]['标准差']
            ax.errorbar(i, mean_reward, yerr=std_reward, fmt="none", color="black", capsize=5)
        
        plt.title("不同环境下各算法的平均奖励对比")
        plt.ylabel("平均奖励")
        plt.tight_layout()
        
        # 保存图表
        reward_plot_path = os.path.join(self.base_dir, "reward_comparison.png")
        plt.savefig(reward_plot_path)
        
        # 绘制训练时间对比图
        plt.figure(figsize=(12, 6))
        sns.barplot(x="环境", y="训练时间(秒)", hue="算法", data=df)
        plt.title("不同环境下各算法的训练时间对比")
        plt.ylabel("训练时间(秒)")
        plt.tight_layout()
        
        # 保存图表
        time_plot_path = os.path.join(self.base_dir, "time_comparison.png")
        plt.savefig(time_plot_path)
        
        print(f"结果对比图已保存至: {self.base_dir}")
    
    def run_experiment(self, env_ids: List[str], algorithms: List[str], 
                     timesteps_per_env: int = 100000) -> None:
        """
        运行实验
        
        Args:
            env_ids: 环境ID列表
            algorithms: 算法列表
            timesteps_per_env: 每个环境的训练步数
        """
        for env_id in env_ids:
            if env_id not in self.supported_envs:
                print(f"警告: 不支持的环境 {env_id}，跳过")
                continue
                
            for algorithm in algorithms:
                if algorithm not in self.supported_envs[env_id]["algorithms"]:
                    print(f"警告: 环境 {env_id} 不支持算法 {algorithm}，跳过")
                    continue
                
                try:
                    # 训练模型
                    self.train_model(env_id, algorithm, timesteps_per_env)
                    
                    # 记录视频
                    self.record_video(env_id, algorithm)
                except Exception as e:
                    print(f"运行 {env_id} 环境的 {algorithm} 算法时出错: {e}")
        
        # 可视化结果
        self.visualize_training_results()


def main():
    """主函数"""
    # 创建可视化器
    visualizer = RLVisualizer()
    
    # 显示可用环境
    visualizer.list_available_environments()
    
    # 选择要运行的环境和算法
    selected_envs = ["CartPole-v1", "LunarLander-v3"]
    selected_algos = ["PPO", "A2C", "DQN"]
    
    # 运行实验
    print("\n开始运行强化学习实验...")
    visualizer.run_experiment(
        env_ids=selected_envs,
        algorithms=selected_algos,
        timesteps_per_env=50000  # 减少步数以加快演示
    )
    
    print("\n实验完成!")
    print(f"结果保存在: {visualizer.base_dir}")


if __name__ == "__main__":
    main()
