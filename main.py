"""
手写文字识别多算法对比主程序
使用PyTorch和Stable Baselines3进行算法对比，集成WandB实验追踪
"""
import os
import sys
import torch
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from config import TRAINING_CONFIG, SB3_ALGORITHMS, WANDB_CONFIG
from algorithm_manager import AlgorithmManager, PyTorchCNN, StableBaselinesAlgorithm
from result_analyzer import ResultAnalyzer
from handwritten_recognition import load_mnist_data


def setup_environment():
    """设置实验环境"""
    print("设置实验环境...")
    
    # 设置随机种子
    torch.manual_seed(TRAINING_CONFIG['random_seed'])
    np.random.seed(TRAINING_CONFIG['random_seed'])
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用，设备: {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(TRAINING_CONFIG['random_seed'])
    else:
        print("CUDA不可用，使用CPU")
    
    # 初始化WandB
    try:
        wandb.login()
        print("WandB已连接")
    except Exception as e:
        print(f"WandB连接失败: {e}")
        print("将在离线模式下运行")
        os.environ["WANDB_MODE"] = "offline"


def prepare_data():
    """准备数据集"""
    print("\n准备MNIST数据集...")
    
    # 加载MNIST数据
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist_data()
    
    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=TRAINING_CONFIG['validation_split'],
        random_state=TRAINING_CONFIG['random_seed'],
        stratify=y_train_full
    )
    
    print(f"训练集大小: {len(x_train)}")
    print(f"验证集大小: {len(x_val)}")
    print(f"测试集大小: {len(x_test)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_algorithms():
    """创建算法实例"""
    print("\n创建算法实例...")
    
    algorithms = []
    
    # 1. PyTorch CNN
    print("  - 创建PyTorch CNN算法")
    pytorch_config = {
        'epochs': TRAINING_CONFIG['epochs'],
        'batch_size': TRAINING_CONFIG['batch_size'],
        'learning_rate': TRAINING_CONFIG['learning_rate']
    }
    algorithms.append(PyTorchCNN(pytorch_config))
    
    # 2. Stable Baselines3算法
    sb3_algorithms = ['PPO', 'A2C', 'DQN']  # SAC可能在离散动作空间有问题，先排除
    
    for algo_name in sb3_algorithms:
        print(f"  - 创建Stable Baselines3 {algo_name}算法")
        try:
            algorithms.append(StableBaselinesAlgorithm(algo_name, SB3_ALGORITHMS[algo_name]))
        except Exception as e:
            print(f"    创建{algo_name}失败: {e}")
    
    print(f"成功创建 {len(algorithms)} 个算法")
    return algorithms


def run_experiments(algorithms, x_train, y_train, x_val, y_val, x_test, y_test):
    """运行实验"""
    print("\n开始运行实验...")
    
    # 创建算法管理器
    manager = AlgorithmManager()
    
    # 注册所有算法
    for algorithm in algorithms:
        manager.register_algorithm(algorithm)
    
    # 训练所有算法
    print("\n" + "="*60)
    print("开始训练阶段")
    print("="*60)
    manager.train_all(x_train, y_train, x_val, y_val)
    
    # 评估所有算法
    print("\n" + "="*60)
    print("开始评估阶段")
    print("="*60)
    manager.evaluate_all(x_test, y_test)
    
    return manager.get_comparison_results()


def analyze_results(results):
    """分析和可视化结果"""
    print("\n" + "="*60)
    print("结果分析阶段")
    print("="*60)
    
    # 创建结果分析器
    analyzer = ResultAnalyzer(results)
    
    # 打印摘要
    analyzer.print_summary()
    
    # 生成对比图表
    print("\n生成对比图表...")
    analyzer.plot_comparison_charts()
    
    # 保存结果
    print("\n保存结果...")
    analyzer.save_results()
    
    # 打印报告
    print("\n" + "="*60)
    print("详细对比报告")
    print("="*60)
    report = analyzer.generate_comparison_report()
    print(report)
    
    return analyzer


def main():
    """主函数"""
    print("手写文字识别多算法对比项目")
    print("="*60)
    print("使用PyTorch和Stable Baselines3进行算法对比")
    print("集成WandB实验追踪")
    print("="*60)
    
    try:
        # 1. 设置环境
        setup_environment()
        
        # 2. 准备数据
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()
        
        # 3. 创建算法
        algorithms = create_algorithms()
        
        if len(algorithms) == 0:
            print("没有可用的算法，程序退出")
            return
        
        # 4. 运行实验
        results = run_experiments(algorithms, x_train, y_train, x_val, y_val, x_test, y_test)
        
        # 5. 分析结果
        analyzer = analyze_results(results)
        
        print("\n" + "="*60)
        print("实验完成！")
        print("="*60)
        print("结果文件已保存到 'results' 目录")
        print("- comparison_results.csv: 对比结果表格")
        print("- comparison_report.md: 详细对比报告")
        print("- comparison_charts.png: 对比图表")
        print("- detailed_results.json: 完整实验数据")
        
    except KeyboardInterrupt:
        print("\n用户中断实验")
    except Exception as e:
        print(f"\n实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理WandB
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()
