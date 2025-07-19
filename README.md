# 手写数字识别神经网络项目 (PyTorch版本)

这个项目使用PyTorch实现的卷积神经网络(CNN)进行MNIST手写数字的识别。

## 项目结构

```
├── main.py                    # 主程序 - 完整实验流程
├── run_experiment.py          # 快速启动脚本 - 交互式菜单
├── config.py                  # 配置文件 - 算法参数配置
├── algorithm_manager.py       # 算法管理器 - 统一算法接口
├── mnist_env.py              # MNIST环境适配器 - RL环境封装
├── result_analyzer.py        # 结果分析器 - 可视化和报告生成
├── handwritten_recognition.py # 原始CNN实现
├── requirements.txt          # 依赖列表
├── dataset/                  # 数据集目录
├── models/                   # 模型保存目录
├── results/                  # 结果输出目录
└── logs/                     # 日志目录
```

## 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd aifresh

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖

- `torch>=2.0.0` - PyTorch深度学习框架
- `stable-baselines3>=2.0.0` - 强化学习算法库
- `wandb>=0.15.0` - 实验追踪工具
- `matplotlib>=3.5.0` - 数据可视化
- `scikit-learn>=1.1.0` - 机器学习工具

## 使用方法

### 方式1: 交互式启动 (推荐)

```bash
python run_experiment.py
```

启动交互式菜单，可选择：
1. 快速实验 (仅PyTorch CNN, 2-3分钟)
2. 完整实验 (所有算法对比, 15-30分钟)
3. 查看项目结构
4. 安装依赖

### 方式2: 命令行启动

```bash
# 快速实验
python run_experiment.py quick

# 完整实验
python run_experiment.py full

# 或直接运行主程序
python main.py
```

### 方式3: 单独运行原始CNN

```bash
python handwritten_recognition.py
```

## 算法说明

### 1. PyTorch CNN
- **类型**: 深度学习卷积神经网络
- **结构**: 3层卷积 + 2层全连接
- **特点**: 直接监督学习，适合图像分类

训练完成后，您可以使用`predict.py`脚本来识别新的手写数字图像：

```bash
python predict.py
```

按照提示输入手写数字图像的路径，程序将显示预测结果。

### 注意事项

- 输入图像应为手写数字，最好是黑底白字或白底黑字的图像
- 图像会被自动调整为28x28像素并进行预处理
- 为获得最佳结果，请确保数字在图像中居中且清晰可见

## 模型结构

该项目使用了一个三层卷积神经网络，结构如下：

1. 卷积层 (32个3x3滤波器) + ReLU激活 + 最大池化
2. 卷积层 (64个3x3滤波器) + ReLU激活 + 最大池化
3. 卷积层 (128个3x3滤波器) + ReLU激活 + 最大池化
4. 全连接层 (128个神经元) + ReLU激活 + Dropout(0.5)
5. 输出层 (10个神经元，对应0-9十个数字) + LogSoftmax激活

## 性能

在MNIST测试集上，该模型通常可以达到约99%的准确率。
