import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np

class FaceEncoder(nn.Module):
    """人脸特征编码器"""
    def __init__(self, embedding_dim=128):
        super(FaceEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNetwork(nn.Module):
    """Siamese网络用于人脸比对"""
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = FaceEncoder(embedding_dim)

    def forward(self, img1, img2):
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        return feat1, feat2

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        distance = F.pairwise_distance(feat1, feat2)
        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss)

class FaceVerificationModel:
    """人脸比对模型封装"""
    def __init__(self, model_path='face_verification_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseNetwork().to(self.device)
        self.model_path = model_path
        self.threshold = 0.5

        if os.path.exists(model_path):
            self.load_model()
            print(f"已加载预训练模型: {model_path}")
        else:
            print("未找到预训练模型，使用随机初始化")

    def load_model(self):
        """加载预训练模型"""
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"加载模型失败: {e}")

    def save_model(self):
        """保存模型"""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"模型已保存到: {self.model_path}")

    def preprocess_image(self, image_path):
        """预处理图像"""
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            return img
        except Exception as e:
            print(f"预处理图像失败: {e}")
            return None

    def compare_faces(self, image_path1, image_path2):
        """比对两张人脸图片"""
        img1 = self.preprocess_image(image_path1)
        img2 = self.preprocess_image(image_path2)

        if img1 is None or img2 is None:
            return None, None

        with torch.no_grad():
            feat1, feat2 = self.model(img1, img2)
            distance = F.pairwise_distance(feat1, feat2).item()
            similarity = 1 - distance

            if distance < self.threshold:
                result = 1
            else:
                result = 0

        return result, similarity

def create_sample_dataset(photo_dir='photos', output_dir='face_pairs'):
    """创建人脸比对数据集。

    生成负样本对（不同照片=非同一个人，label=0）。
    正样本对（同一个人不同照片，label=1）需要人工标注。
    自比对 (photo_i, photo_i) 不生成——对训练无意义。
    """
    os.makedirs(output_dir, exist_ok=True)

    photos = [f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    pairs = []
    labels = []

    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            pairs.append((photos[i], photos[j]))
            labels.append(0)  # 默认：不同人

    np.save(os.path.join(output_dir, 'pairs.npy'), np.asarray(pairs))
    np.save(os.path.join(output_dir, 'labels.npy'), np.asarray(labels))

    print(f"数据集创建完成: {len(pairs)} 对图片（全部为负样本）")
    print("注意：正样本（同一人不同照片）需人工标注 labels 数组。")
    return pairs, labels

if __name__ == '__main__':
    model = FaceVerificationModel()

    example1 = 'photos/photo1.jpg'
    example2 = 'photos/photo2.jpg'

    if os.path.exists(example1) and os.path.exists(example2):
        result, similarity = model.compare_faces(example1, example2)
        if result is not None:
            print(f"比对结果: {'同一个人' if result == 1 else '不是同一个人'}")
            print(f"相似度: {similarity:.4f}")
        else:
            print("比对失败")
    else:
        print("请确保示例图片存在")
