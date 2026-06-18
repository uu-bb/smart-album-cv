import os
import cv2
import numpy as np

# 表情标签（二分类：有笑容/无笑容）
EMOTIONS = ['no_smile', 'smile']

# 尝试导入PyTorch相关模块
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as transforms
    import cv2
    import numpy as np
    TORCH_AVAILABLE = True

    # 共享预处理参数（与 predict_emotion 保持一致）
    _PREPROC_MEAN = (0.485, 0.456, 0.406)
    _PREPROC_STD = (0.229, 0.224, 0.225)
    _INPUT_SIZE = 64

    def _preprocess_from_path(img_path):
        """统一的图像预处理：从路径加载并转为归一化张量"""
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(_PREPROC_MEAN, dtype=torch.float32)
        std = torch.tensor(_PREPROC_STD, dtype=torch.float32)
        tensor = (tensor - mean[:, None, None]) / std[:, None, None]
        return tensor

    class EmotionDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.images = []
            self.labels = []

            for label_idx, emotion in enumerate(EMOTIONS):
                emotion_dir = os.path.join(data_dir, emotion)
                if os.path.exists(emotion_dir):
                    for img_name in os.listdir(emotion_dir):
                        img_path = os.path.join(emotion_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(label_idx)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]
            try:
                tensor = _preprocess_from_path(img_path)
                return tensor, label
            except Exception as e:
                print(f"加载图像失败: {e}")
                return torch.randn(3, _INPUT_SIZE, _INPUT_SIZE), label

    class EmotionClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super(EmotionClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def train_model(data_dir, epochs=20, batch_size=32, learning_rate=0.001):
        try:
            # 创建数据集
            dataset = EmotionDataset(data_dir)

            # 显示数据集信息
            print(f'数据集大小: {len(dataset)}')
            print(f'图像路径: {dataset.images[:5]}')  # 显示前5个图像路径
            print(f'标签: {dataset.labels[:5]}')  # 显示前5个标签

            # 检查数据集大小
            if len(dataset) == 0:
                print('错误：数据集为空，请确保在data/smile_dataset目录下添加足够的图像')
                print('需要在no_smile和smile目录下都添加图像')
                return None

            # 检查是否有至少一个类别的数据
            unique_labels = set(dataset.labels)
            if len(unique_labels) < 1:
                print('错误：数据集为空，请确保在data/smile_dataset目录下添加足够的图像')
                return None
            elif len(unique_labels) < 2:
                print('警告：数据集只有一个类别的数据，训练可能效果不佳')
                print(f'当前标签: {unique_labels}')
                # 继续训练，不返回None

            # 划分训练集和测试集
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            # 确保至少有一个样本用于训练和测试
            if train_size < 1 or test_size < 1:
                print('错误：数据集太小，需要更多的图像')
                return None

            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 初始化模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = EmotionClassifier().to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练模型
            train_losses = []
            test_accuracies = []

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0

                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                train_loss = running_loss / len(train_loader)
                train_losses.append(train_loss)

                # 测试模型
                model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                test_accuracies.append(accuracy)

                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # 保存模型
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/emotion_classifier.pth')
            print('模型已保存到 models/emotion_classifier.pth')

            # 绘制训练曲线
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Training Loss')
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(test_accuracies, label='Test Accuracy')
                plt.title('Test Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()

                plt.tight_layout()
                plt.savefig('training_results.png')
                print('训练结果已保存到 training_results.png')
            except Exception as e:
                print(f"绘制训练曲线失败: {e}")
                print('跳过训练曲线绘制步骤')

            # 保存训练历史数据到文件
            try:
                import json
                training_history = {
                    'train_losses': train_losses,
                    'test_accuracies': test_accuracies,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
                with open('training_history.json', 'w') as f:
                    json.dump(training_history, f, indent=2)
                print('训练历史数据已保存到 training_history.json')
            except Exception as e:
                print(f"保存训练历史数据失败: {e}")

            return model
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_model(model_path):
        model = EmotionClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_emotion(model, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        tensor = _preprocess_from_path(image_path).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs.data, 1)
            emotion = EMOTIONS[predicted.item()]

        return emotion
except ImportError:
    print("PyTorch模块未安装，无法进行模型训练")
    print("请运行: pip install torch torchvision")
    TORCH_AVAILABLE = False

    # 创建占位函数
    def train_model(data_dir, epochs=20, batch_size=32, learning_rate=0.001):
        print("PyTorch未安装，无法进行模型训练")
        return None

    def load_model(model_path):
        print("PyTorch未安装，无法加载模型")
        return None

    def predict_emotion(model, image_path):
        print("PyTorch未安装，无法进行预测")
        return None

if __name__ == '__main__':
    if TORCH_AVAILABLE:
        # 示例用法
        # 假设数据集在 'data/smile_dataset' 目录下
        data_dir = 'data/smile_dataset'
        if os.path.exists(data_dir):
            model = train_model(data_dir)
        else:
            print(f'数据集目录 {data_dir} 不存在，请创建并添加笑容分类数据')
            print('请按照以下结构组织数据:')
            print('data/smile_dataset/')
            print('├── no_smile/  # 无笑容的图片')
            print('└── smile/     # 有笑容的图片')
            print('建议使用包含笑容和非笑容的人脸数据集')
    else:
        print('PyTorch未安装，无法进行模型训练')
        print('请安装PyTorch后再运行')
