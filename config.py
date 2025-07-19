"""
项目配置文件
"""
import os

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')

# 确保目录存在
for path in [DATASET_PATH, MODEL_PATH, RESULTS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)

# 训练配置
TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 128,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'random_seed': 42,
    'device': 'cuda' if os.environ.get('CUDA_AVAILABLE', 'True') == 'True' else 'cpu'
}

# Stable Baselines算法配置
SB3_ALGORITHMS = {
    'PPO': {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'total_timesteps': 100000
    },
    'A2C': {
        'policy': 'MlpPolicy',
        'learning_rate': 7e-4,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.01,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'total_timesteps': 100000
    },
    'DQN': {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'total_timesteps': 100000
    },
    'SAC': {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'total_timesteps': 100000
    }
}

# WandB配置
WANDB_CONFIG = {
    'project': 'handwritten-recognition-comparison',
    'entity': None,  # 用户可以设置自己的entity
    'tags': ['pytorch', 'stable-baselines3', 'mnist', 'comparison'],
    'notes': '多算法手写文字识别对比实验'
}

# 评估指标
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'training_time',
    'inference_time',
    'model_size'
]
