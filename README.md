# 手写数字识别神经网络项目 (PyTorch版本)

这个项目使用PyTorch实现的卷积神经网络(CNN)进行MNIST手写数字的识别。

## 项目结构

```
.
├── dataset/                  # 存放下载的MNIST数据集
├── models/                   # 存放训练好的模型
├── handwritten_recognition.py # 主程序：数据处理、模型训练和评估
├── predict.py                # 预测脚本：用于识别新的手写数字图像
└── README.md                 # 项目说明文档
```

## 环境要求

项目需要以下Python库：

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.5
matplotlib>=3.3.4
pillow>=8.1.0
```

您可以使用以下命令安装所需依赖：

```bash
pip install torch torchvision numpy matplotlib pillow
```

## 使用方法

### 1. 训练模型

运行主程序来下载MNIST数据集并训练模型：

```bash
python handwritten_recognition.py
```

这将会：
- 下载MNIST数据集并保存到本地`dataset`目录
- 创建并训练CNN模型
- 在测试集上评估模型性能
- 保存训练好的模型到`models`目录
- 生成训练历史图表和预测结果图表

### 2. 预测新图像

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
