import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

def load_and_preprocess_image(image_path):
    """
    加载并预处理图像用于预测
    """
    # 加载图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    
    # 调整大小为28x28
    img = img.resize((28, 28))
    
    # 反转颜色（MNIST数据集是黑底白字）
    img = ImageOps.invert(img)
    
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 归一化
    img_array = img_array.astype('float32') / 255.0
    
    # 调整形状以适应PyTorch模型 (1, 1, 28, 28)
    img_array = img_array.reshape(1, 1, 28, 28)
    
    # 转换为PyTorch张量
    img_tensor = torch.FloatTensor(img_array)
    
    return img_tensor

def predict_digit(model, image_tensor, device):
    """
    使用模型预测数字
    """
    # 将图像移至设备
    image_tensor = image_tensor.to(device)
    
    # 设置为评估模式
    model.eval()
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        
    # 获取预测类别
    _, predicted_class = output.max(1)
    predicted_class = predicted_class.item()
    
    # 获取预测概率
    probabilities = torch.exp(output)
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    """
    主函数
    """
    print("手写数字识别 - 预测脚本")
    print("-" * 40)
    
    # 定义CNN模型类
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 第一个卷积层
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            
            # 第二个卷积层
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            
            # 第三个卷积层
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            
            # 全连接层
            self.fc1 = nn.Linear(128, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            # 第一个卷积块
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            
            # 第二个卷积块
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            
            # 第三个卷积块
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            
            # 展平
            x = x.view(x.size(0), -1)
            
            # 全连接层
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return F.log_softmax(x, dim=1)
    
    # 模型路径
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_model.pth')
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在于 {model_path}")
        print("请先运行 handwritten_recognition.py 训练模型")
        return
    
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型实例
    print("加载模型...")
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("模型加载成功!")
    
    # 获取用户输入的图像路径
    while True:
        image_path = input("\n请输入手写数字图像的路径 (输入 'q' 退出): ")
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"错误: 文件 {image_path} 不存在")
            continue
        
        try:
            # 加载并预处理图像
            img_tensor = load_and_preprocess_image(image_path)
            
            # 预测
            predicted_digit, confidence = predict_digit(model, img_tensor, device)
            
            # 显示结果
            print(f"预测结果: {predicted_digit}")
            print(f"置信度: {confidence:.4f}")
            
            # 显示图像
            plt.figure(figsize=(4, 4))
            plt.imshow(img_tensor.numpy().reshape(28, 28), cmap='gray')
            plt.title(f"预测: {predicted_digit} (置信度: {confidence:.4f})")
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"预测过程中出错: {e}")

if __name__ == "__main__":
    main()
