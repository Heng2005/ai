"""
快速启动实验脚本
简化版本，适合快速测试和演示
"""
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from config import TRAINING_CONFIG
from algorithm_manager import AlgorithmManager, PyTorchCNN
from result_analyzer import ResultAnalyzer
from handwritten_recognition import load_mnist_data
from sklearn.model_selection import train_test_split


def quick_experiment():
    """快速实验 - 只运行PyTorch CNN进行基础测试"""
    print("快速实验模式 - 仅测试PyTorch CNN")
    print("="*50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    print("加载MNIST数据...")
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist_data()
    
    # 使用较小的数据集进行快速测试
    x_train_small = x_train_full[:5000]  # 只用5000个训练样本
    y_train_small = y_train_full[:5000]
    x_test_small = x_test[:1000]  # 只用1000个测试样本
    y_test_small = y_test[:1000]
    
    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_small, y_train_small, 
        test_size=0.2, random_state=42, stratify=y_train_small
    )
    
    print(f"训练集: {len(x_train)}, 验证集: {len(x_val)}, 测试集: {len(x_test_small)}")
    
    # 创建PyTorch CNN算法
    config = {
        'epochs': 5,  # 减少训练轮数
        'batch_size': 128,
        'learning_rate': 0.001
    }
    
    algorithm = PyTorchCNN(config)
    
    # 训练
    print("\n开始训练...")
    history = algorithm.train(x_train, y_train, x_val, y_val)
    
    # 评估
    print("\n开始评估...")
    metrics = algorithm.evaluate(x_test_small, y_test_small)
    
    # 显示结果
    print("\n" + "="*50)
    print("实验结果")
    print("="*50)
    print(f"测试准确率: {metrics['accuracy']:.2f}%")
    print(f"训练时间: {metrics['training_time']:.2f}秒")
    print(f"推理时间: {metrics['inference_time']:.4f}秒")
    print(f"模型大小: {metrics['model_size_mb']:.2f}MB")
    
    return metrics


def full_experiment():
    """完整实验 - 运行所有算法对比"""
    print("完整实验模式 - 运行所有算法对比")
    print("="*50)
    print("注意: 这可能需要较长时间...")
    
    # 导入主程序并运行
    from main import main
    main()


def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n" + "="*60)
        print("手写文字识别多算法对比项目")
        print("="*60)
        print("请选择运行模式:")
        print("1. 快速实验 (仅PyTorch CNN, 约2-3分钟)")
        print("2. 完整实验 (所有算法对比, 约15-30分钟)")
        print("3. 查看项目结构")
        print("4. 安装依赖")
        print("5. 退出")
        print("="*60)
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            try:
                quick_experiment()
                print("\n快速实验完成！")
            except Exception as e:
                print(f"实验失败: {e}")
                
        elif choice == '2':
            try:
                full_experiment()
                print("\n完整实验完成！")
            except Exception as e:
                print(f"实验失败: {e}")
                
        elif choice == '3':
            show_project_structure()
            
        elif choice == '4':
            install_dependencies()
            
        elif choice == '5':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")


def show_project_structure():
    """显示项目结构"""
    print("\n项目结构:")
    print("="*40)
    print("├── main.py                    # 主程序")
    print("├── run_experiment.py          # 快速启动脚本")
    print("├── config.py                  # 配置文件")
    print("├── algorithm_manager.py       # 算法管理器")
    print("├── mnist_env.py              # MNIST环境适配器")
    print("├── result_analyzer.py        # 结果分析器")
    print("├── handwritten_recognition.py # 原始CNN实现")
    print("├── requirements.txt          # 依赖列表")
    print("├── dataset/                  # 数据集目录")
    print("├── models/                   # 模型保存目录")
    print("├── results/                  # 结果输出目录")
    print("└── logs/                     # 日志目录")
    print("="*40)
    
    print("\n核心功能:")
    print("- PyTorch CNN: 深度学习卷积神经网络")
    print("- Stable Baselines3: 强化学习算法 (PPO, A2C, DQN)")
    print("- WandB集成: 实验追踪和可视化")
    print("- 统一评估: 准确率、训练时间、推理时间对比")
    print("- 结果分析: 自动生成报告和图表")


def install_dependencies():
    """安装依赖"""
    print("\n安装项目依赖...")
    print("请在命令行中运行以下命令:")
    print("pip install -r requirements.txt")
    print("\n主要依赖包括:")
    print("- torch, torchvision: PyTorch深度学习框架")
    print("- stable-baselines3: 强化学习算法库")
    print("- wandb: 实验追踪工具")
    print("- matplotlib, seaborn: 数据可视化")
    print("- scikit-learn: 机器学习工具")


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_experiment()
        elif sys.argv[1] == "full":
            full_experiment()
        else:
            print("用法: python run_experiment.py [quick|full]")
    else:
        # 启动交互式菜单
        interactive_menu()
