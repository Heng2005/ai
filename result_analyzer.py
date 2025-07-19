"""
结果分析和可视化模块
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List
import os
from config import RESULTS_PATH
import json


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.comparison_df = None
        self._prepare_comparison_data()
        
    def _prepare_comparison_data(self):
        """准备对比数据"""
        data = []
        
        for algorithm_name, result in self.results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                row = {
                    'Algorithm': algorithm_name,
                    'Accuracy (%)': metrics.get('accuracy', 0),
                    'Training Time (s)': metrics.get('training_time', 0),
                    'Inference Time (s)': metrics.get('inference_time', 0),
                    'Model Size (MB)': metrics.get('model_size_mb', 0)
                }
                data.append(row)
        
        self.comparison_df = pd.DataFrame(data)
        
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        if self.comparison_df is None or len(self.comparison_df) == 0:
            return "没有可用的对比数据"
        
        report = []
        report.append("# 手写文字识别算法对比报告\n")
        
        # 总体统计
        report.append("## 总体性能对比\n")
        report.append(self.comparison_df.to_string(index=False))
        report.append("\n\n")
        
        # 最佳性能分析
        report.append("## 最佳性能分析\n")
        
        best_accuracy = self.comparison_df.loc[self.comparison_df['Accuracy (%)'].idxmax()]
        report.append(f"**最高准确率**: {best_accuracy['Algorithm']} - {best_accuracy['Accuracy (%)']:.2f}%\n")
        
        fastest_training = self.comparison_df.loc[self.comparison_df['Training Time (s)'].idxmin()]
        report.append(f"**最快训练**: {fastest_training['Algorithm']} - {fastest_training['Training Time (s)']:.2f}秒\n")
        
        fastest_inference = self.comparison_df.loc[self.comparison_df['Inference Time (s)'].idxmin()]
        report.append(f"**最快推理**: {fastest_inference['Algorithm']} - {fastest_inference['Inference Time (s)']:.4f}秒\n")
        
        if self.comparison_df['Model Size (MB)'].sum() > 0:
            smallest_model = self.comparison_df.loc[self.comparison_df['Model Size (MB)'].idxmin()]
            report.append(f"**最小模型**: {smallest_model['Algorithm']} - {smallest_model['Model Size (MB)']:.2f}MB\n")
        
        report.append("\n")
        
        # 算法特点分析
        report.append("## 算法特点分析\n")
        
        for _, row in self.comparison_df.iterrows():
            algorithm = row['Algorithm']
            accuracy = row['Accuracy (%)']
            train_time = row['Training Time (s)']
            
            report.append(f"### {algorithm}\n")
            
            if 'PyTorch' in algorithm:
                report.append("- **类型**: 深度学习 (CNN)\n")
                report.append("- **特点**: 直接监督学习，适合图像分类任务\n")
            elif 'SB3' in algorithm:
                report.append("- **类型**: 强化学习\n")
                if 'PPO' in algorithm:
                    report.append("- **特点**: 策略优化算法，稳定性好\n")
                elif 'A2C' in algorithm:
                    report.append("- **特点**: Actor-Critic算法，收敛快\n")
                elif 'DQN' in algorithm:
                    report.append("- **特点**: 深度Q网络，适合离散动作空间\n")
                elif 'SAC' in algorithm:
                    report.append("- **特点**: 软Actor-Critic，样本效率高\n")
            
            report.append(f"- **准确率**: {accuracy:.2f}%\n")
            report.append(f"- **训练时间**: {train_time:.2f}秒\n")
            
            # 性能评价
            if accuracy >= 95:
                report.append("- **性能评价**: 优秀\n")
            elif accuracy >= 90:
                report.append("- **性能评价**: 良好\n")
            elif accuracy >= 80:
                report.append("- **性能评价**: 一般\n")
            else:
                report.append("- **性能评价**: 需要改进\n")
            
            report.append("\n")
        
        # 推荐建议
        report.append("## 推荐建议\n")
        
        best_algo = self.comparison_df.loc[self.comparison_df['Accuracy (%)'].idxmax(), 'Algorithm']
        report.append(f"1. **准确率优先**: 推荐使用 {best_algo}\n")
        
        fastest_algo = self.comparison_df.loc[self.comparison_df['Training Time (s)'].idxmin(), 'Algorithm']
        report.append(f"2. **训练速度优先**: 推荐使用 {fastest_algo}\n")
        
        report.append("3. **实际应用建议**:\n")
        report.append("   - 对于生产环境，建议使用PyTorch CNN，准确率高且推理速度快\n")
        report.append("   - 对于研究目的，可以尝试不同的强化学习算法\n")
        report.append("   - 对于资源受限环境，考虑模型大小和推理时间\n")
        
        return "".join(report)
    
    def plot_comparison_charts(self):
        """绘制对比图表"""
        if self.comparison_df is None or len(self.comparison_df) == 0:
            print("没有可用的对比数据")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('手写文字识别算法对比分析', fontsize=16, fontweight='bold')
        
        # 1. 准确率对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(self.comparison_df['Algorithm'], self.comparison_df['Accuracy (%)'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('准确率对比', fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 旋转x轴标签
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 训练时间对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(self.comparison_df['Algorithm'], self.comparison_df['Training Time (s)'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title('训练时间对比', fontweight='bold')
        ax2.set_ylabel('训练时间 (秒)')
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(self.comparison_df['Training Time (s)']) * 0.01,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 推理时间对比
        ax3 = axes[1, 0]
        bars3 = ax3.bar(self.comparison_df['Algorithm'], self.comparison_df['Inference Time (s)'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax3.set_title('推理时间对比', fontweight='bold')
        ax3.set_ylabel('推理时间 (秒)')
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(self.comparison_df['Inference Time (s)']) * 0.01,
                    f'{height:.4f}s', ha='center', va='bottom')
        
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 综合性能雷达图
        ax4 = axes[1, 1]
        
        # 准备雷达图数据（标准化到0-1）
        metrics = ['Accuracy', 'Speed (1/Training Time)', 'Inference Speed', 'Efficiency']
        
        # 标准化数据
        normalized_data = []
        for _, row in self.comparison_df.iterrows():
            acc_norm = row['Accuracy (%)'] / 100
            speed_norm = 1 / (1 + row['Training Time (s)'] / max(self.comparison_df['Training Time (s)']))
            inf_norm = 1 / (1 + row['Inference Time (s)'] / max(self.comparison_df['Inference Time (s)']))
            eff_norm = (acc_norm + speed_norm + inf_norm) / 3
            
            normalized_data.append([acc_norm, speed_norm, inf_norm, eff_norm])
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (_, row) in enumerate(self.comparison_df.iterrows()):
            values = normalized_data[i] + normalized_data[i][:1]  # 闭合
            ax4.plot(angles, values, 'o-', linewidth=2, label=row['Algorithm'], color=colors[i % len(colors)])
            ax4.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('综合性能对比', fontweight='bold')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(RESULTS_PATH, 'comparison_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存到: {chart_path}")
        
        plt.show()
    
    def save_results(self):
        """保存结果到文件"""
        # 保存CSV
        if self.comparison_df is not None:
            csv_path = os.path.join(RESULTS_PATH, 'comparison_results.csv')
            self.comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"对比结果已保存到: {csv_path}")
        
        # 保存详细结果JSON
        json_path = os.path.join(RESULTS_PATH, 'detailed_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {json_path}")
        
        # 保存报告
        report = self.generate_comparison_report()
        report_path = os.path.join(RESULTS_PATH, 'comparison_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"对比报告已保存到: {report_path}")
    
    def print_summary(self):
        """打印摘要"""
        if self.comparison_df is None or len(self.comparison_df) == 0:
            print("没有可用的对比数据")
            return
        
        print("\n" + "="*60)
        print("手写文字识别算法对比摘要")
        print("="*60)
        
        print(f"\n共测试了 {len(self.comparison_df)} 种算法:")
        for algorithm in self.comparison_df['Algorithm']:
            print(f"  - {algorithm}")
        
        print(f"\n最佳性能:")
        best_acc_idx = self.comparison_df['Accuracy (%)'].idxmax()
        best_acc_algo = self.comparison_df.loc[best_acc_idx, 'Algorithm']
        best_acc_value = self.comparison_df.loc[best_acc_idx, 'Accuracy (%)']
        print(f"  准确率最高: {best_acc_algo} ({best_acc_value:.2f}%)")
        
        fastest_idx = self.comparison_df['Training Time (s)'].idxmin()
        fastest_algo = self.comparison_df.loc[fastest_idx, 'Algorithm']
        fastest_time = self.comparison_df.loc[fastest_idx, 'Training Time (s)']
        print(f"  训练最快: {fastest_algo} ({fastest_time:.2f}秒)")
        
        print("\n详细结果请查看生成的报告和图表。")
        print("="*60)
