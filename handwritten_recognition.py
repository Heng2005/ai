import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 数据集路径
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# 确保目录存在
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

def load_mnist_data():
    """
    加载MNIST数据集
    """
    print("正在加载MNIST数据集...")
    
    # 检查本地是否已有数据集
    if (os.path.exists(os.path.join(DATASET_PATH, 'mnist_train_images.npy')) and
        os.path.exists(os.path.join(DATASET_PATH, 'mnist_train_labels.npy')) and
        os.path.exists(os.path.join(DATASET_PATH, 'mnist_test_images.npy')) and
        os.path.exists(os.path.join(DATASET_PATH, 'mnist_test_labels.npy'))):
        
        print("从本地加载数据集...")
        x_train = np.load(os.path.join(DATASET_PATH, 'mnist_train_images.npy'))
        y_train = np.load(os.path.join(DATASET_PATH, 'mnist_train_labels.npy'))
        x_test = np.load(os.path.join(DATASET_PATH, 'mnist_test_images.npy'))
        y_test = np.load(os.path.join(DATASET_PATH, 'mnist_test_labels.npy'))
    else:
        print("从torchvision下载MNIST数据集...")
        # 定义转换
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 下载训练集
        train_dataset = datasets.MNIST(root=DATASET_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=DATASET_PATH, train=False, download=True, transform=transform)
        
        # 提取数据和标签
        x_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        x_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()
        
        # 保存到本地
        print("保存数据集到本地...")
        np.save(os.path.join(DATASET_PATH, 'mnist_train_images.npy'), x_train)
        np.save(os.path.join(DATASET_PATH, 'mnist_train_labels.npy'), y_train)
        np.save(os.path.join(DATASET_PATH, 'mnist_test_images.npy'), x_test)
        np.save(os.path.join(DATASET_PATH, 'mnist_test_labels.npy'), y_test)
        print("数据集已保存到本地")
    
    # 数据预处理
    # 将图像数据归一化到 [0, 1] 范围
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 调整数据形状以适应PyTorch CNN (样本数, 通道数, 高度, 宽度)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    
    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    
    print(f"训练数据形状: {x_train.shape}")
    print(f"测试数据形状: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

class CNN(nn.Module):
    """
    创建CNN模型用于手写数字识别
    """
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

def create_model():
    """
    创建并返回CNN模型
    """
    model = CNN()
    
    # 打印模型结构
    print(model)
    
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 将模型移至设备
    model = model.to(device)
    
    return model, device

def train_model(model, device, x_train, y_train, x_test, y_test):
    """
    训练模型
    """
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 训练参数
    epochs = 10
    best_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移至设备
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
        
        # 计算训练集平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # 评估模式
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # 将数据移至设备
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                output = model(data)
                
                # 计算损失
                loss = criterion(output, target)
                
                # 累计损失
                val_loss += loss.item()
                
                # 计算准确率
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # 计算验证集平均损失和准确率
        val_loss = val_loss / len(test_loader)
        val_accuracy = 100. * correct / total
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            print(f'验证准确率提高 ({best_accuracy:.2f}% --> {val_accuracy:.2f}%)，保存模型...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'best_model.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'final_model.pth'))
    print(f"训练完成，最佳验证准确率: {best_accuracy:.2f}%")
    
    return history

def evaluate_model(model, device, x_test, y_test):
    """
    评估模型性能
    """
    # 创建测试数据加载器
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 评估模式
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    # 存储一些预测结果用于可视化
    all_preds = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移至设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 累计损失
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 存储预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_images.extend(data.cpu().numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    
    print(f"测试集准确率: {test_accuracy:.2f}%")
    print(f"测试集损失: {test_loss:.4f}")
    
    # 显示一些预测结果
    plt.figure(figsize=(12, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(all_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"预测: {all_preds[i]}, 实际: {all_targets[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'predictions.png'))
    plt.show()

def plot_training_history(history):
    """
    绘制训练历史
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, 'training_history.png'))
    plt.show()

def main():
    """
    主函数
    """
    print("手写数字识别 - 神经网络项目 (PyTorch版本)")
    print("-" * 40)
    
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # 创建模型
    model, device = create_model()
    
    # 训练模型
    history = train_model(model, device, x_train, y_train, x_test, y_test)
    
    # 评估模型
    evaluate_model(model, device, x_test, y_test)
    
    # 绘制训练历史
    plot_training_history(history)

if __name__ == "__main__":
    main()
