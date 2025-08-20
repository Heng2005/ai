"""
强化学习实验模块 - 用于运行和可视化强化学习实验
"""
import os
import sys
import time
import argparse
from typing import List, Dict, Any

# 导入项目模块
from rl_visualizer import RLVisualizer


def run_rl_experiment(envs: List[str] = None, 
                     algorithms: List[str] = None, 
                     timesteps: int = 50000,
                     video_length: int = 1000):
    """
    运行强化学习实验
    
    Args:
        envs: 要使用的环境列表，默认为None（使用默认环境）
        algorithms: 要使用的算法列表，默认为None（使用默认算法）
        timesteps: 每个环境的训练步数
        video_length: 录制视频的长度
    """
    # 创建可视化器
    visualizer = RLVisualizer()
    
    # 如果未指定环境，使用默认环境
    if envs is None:
        envs = ["CartPole-v1", "LunarLander-v3"]
    
    # 如果未指定算法，使用默认算法
    if algorithms is None:
        algorithms = ["PPO", "A2C", "DQN"]
    
    # 显示可用环境
    visualizer.list_available_environments()
    
    # 确认选择
    print(f"\n选择的环境: {', '.join(envs)}")
    print(f"选择的算法: {', '.join(algorithms)}")
    print(f"每个环境训练步数: {timesteps}")
    
    # 运行实验
    print("\n开始运行强化学习实验...")
    start_time = time.time()
    
    visualizer.run_experiment(
        env_ids=envs,
        algorithms=algorithms,
        timesteps_per_env=timesteps
    )
    
    total_time = time.time() - start_time
    print(f"\n实验完成! 总耗时: {total_time:.2f}秒")
    print(f"结果保存在: {visualizer.base_dir}")
    
    return visualizer.training_results


def interactive_rl_menu():
    """交互式强化学习菜单"""
    visualizer = RLVisualizer()
    
    while True:
        print("\n" + "="*60)
        print("强化学习可视化系统")
        print("="*60)
        print("请选择操作:")
        print("1. 查看可用环境")
        print("2. 运行单个环境实验")
        print("3. 运行多环境对比实验")
        print("4. 可视化已有结果")
        print("5. 返回主菜单")
        print("="*60)
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            visualizer.list_available_environments()
                
        elif choice == '2':
            # 单个环境实验
            visualizer.list_available_environments()
            env_id = input("\n请输入环境ID (例如 CartPole-v1): ").strip()
            
            if env_id not in visualizer.supported_envs:
                print(f"错误: 不支持的环境 {env_id}")
                continue
                
            print(f"\n环境 {env_id} 支持的算法: {', '.join(visualizer.supported_envs[env_id]['algorithms'])}")
            algo = input("请输入算法名称 (例如 PPO): ").strip()
            
            if algo not in visualizer.supported_envs[env_id]["algorithms"]:
                print(f"错误: 环境 {env_id} 不支持算法 {algo}")
                continue
                
            try:
                timesteps = int(input("请输入训练步数 (默认 50000): ").strip() or "50000")
                
                # 训练模型
                model, results = visualizer.train_model(env_id, algo, timesteps)
                
                # 询问是否录制视频
                if input("是否录制视频? (y/n): ").strip().lower() == 'y':
                    visualizer.record_video(env_id, algo)
            except Exception as e:
                print(f"实验失败: {e}")
                
        elif choice == '3':
            # 多环境对比实验
            visualizer.list_available_environments()
            
            env_input = input("\n请输入环境ID列表，用逗号分隔 (例如 CartPole-v1,LunarLander-v2): ").strip()
            env_ids = [e.strip() for e in env_input.split(",") if e.strip()]
            
            algo_input = input("请输入算法列表，用逗号分隔 (例如 PPO,A2C,DQN): ").strip()
            algos = [a.strip() for a in algo_input.split(",") if a.strip()]
            
            try:
                timesteps = int(input("请输入每个环境的训练步数 (默认 50000): ").strip() or "50000")
                
                # 运行实验
                run_rl_experiment(env_ids, algos, timesteps)
            except Exception as e:
                print(f"实验失败: {e}")
                
        elif choice == '4':
            # 可视化已有结果
            if not visualizer.training_results:
                print("没有训练结果可供可视化。请先运行实验。")
            else:
                visualizer.visualize_training_results()
                
        elif choice == '5':
            print("返回主菜单")
            break
            
        else:
            print("无效选择，请重新输入")


def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    # 检查tensorboard
    try:
        import tensorboard
    except ImportError:
        missing_deps.append("tensorboard")
    
    # 检查Box2D
    try:
        import gymnasium
        try:
            gymnasium.make("LunarLander-v3")
        except ImportError:
            missing_deps.append("gymnasium[box2d]")
            print("\n" + "="*60)
            print("警告: Box2D依赖缺失，无法运行LunarLander和BipedalWalker环境")
            print("请运行以下命令安装:")
            print("pip install swig")
            print('pip install "gymnasium[box2d]"')
            print("="*60 + "\n")
    except Exception as e:
        print(f"检查Box2D依赖时出错: {e}")
    
    # 如果有缺失的依赖，显示安装提示
    if missing_deps and "gymnasium[box2d]" not in missing_deps:  # Box2D已单独处理
        print("\n" + "="*60)
        print("警告: 以下依赖缺失，可能会影响功能:")
        for dep in missing_deps:
            if dep != "gymnasium[box2d]":
                print(f"- {dep}")
        print("\n如需安装，请运行:")
        deps_to_install = [dep for dep in missing_deps if dep != "gymnasium[box2d]"]
        if deps_to_install:
            print(f"pip install {' '.join(deps_to_install)}")
        print("="*60 + "\n")


def main():
    """主函数"""
    # 检查依赖
    check_dependencies()
    
    parser = argparse.ArgumentParser(description="强化学习实验")
    parser.add_argument("--envs", type=str, default="CartPole-v1,LunarLander-v3",
                        help="要使用的环境列表，用逗号分隔")
    parser.add_argument("--algos", type=str, default="PPO,A2C,DQN",
                        help="要使用的算法列表，用逗号分隔")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="每个环境的训练步数")
    parser.add_argument("--interactive", action="store_true",
                        help="启动交互式菜单")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_rl_menu()
    else:
        envs = [e.strip() for e in args.envs.split(",") if e.strip()]
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
        run_rl_experiment(envs, algos, args.timesteps)


if __name__ == "__main__":
    main()
