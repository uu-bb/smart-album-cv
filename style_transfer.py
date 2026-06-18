import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# 检查PyTorch是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class StyleTransfer:
    def __init__(self):
        if not TORCH_AVAILABLE:
            print("PyTorch未安装，无法使用风格迁移功能")
            return

        # 加载预训练的VGG19模型
        self.vgg = models.vgg19(pretrained=True).features
        # 冻结模型参数
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = self.vgg.to(self.device)

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        """加载并预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)

    def get_features(self, image):
        """提取图像特征"""
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # 内容特征
            '28': 'conv5_1'
        }

        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def gram_matrix(self, tensor):
        """计算Gram矩阵"""
        batch_size, depth, height, width = tensor.size()
        tensor = tensor.view(batch_size * depth, height * width)
        gram = torch.mm(tensor, tensor.t())
        return gram

    def transfer_style(self, content_path, style_path, epochs=300, style_weight=1e6,
                       content_weight=1, progress_callback=None, early_stop_patience=60):
        """执行风格迁移

        Args:
            progress_callback: 可选回调 fn(epoch, total_loss) 供 GUI 进度条使用
            early_stop_patience: 损失变化 < 0.1% 持续此轮数则提前停止（0=禁用）
        """
        if not TORCH_AVAILABLE:
            return None

        content_image = self.load_image(content_path)
        style_image = self.load_image(style_path)

        content_features = self.get_features(content_image)
        style_features = self.get_features(style_image)

        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        generated = content_image.clone().requires_grad_(True)
        optimizer = optim.Adam([generated], lr=0.003)

        style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.75,
            'conv3_1': 0.5,
            'conv4_1': 0.25,
            'conv5_1': 0.1
        }

        best_loss = float('inf')
        stale_count = 0

        for epoch in range(epochs):
            generated_features = self.get_features(generated)

            content_loss = torch.mean(
                (generated_features['conv4_2'] - content_features['conv4_2']) ** 2)

            style_loss = 0
            for layer in style_weights:
                gen_gram = self.gram_matrix(generated_features[layer])
                style_gram = style_grams[layer]
                layer_loss = style_weights[layer] * torch.mean((gen_gram - style_gram) ** 2)
                _, d, h_l, w_l = generated_features[layer].size()
                style_loss += layer_loss / (d * h_l * w_l)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 早停检查
            loss_val = total_loss.item()
            if early_stop_patience > 0:
                if loss_val < best_loss * 0.999:
                    best_loss = loss_val
                    stale_count = 0
                else:
                    stale_count += 1
                    if stale_count >= early_stop_patience:
                        if epoch > 100:  # 至少训练100轮
                            break

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Total Loss: {loss_val:.4f}')

            if progress_callback:
                progress_callback(epoch + 1, epochs)

        # 后处理
        generated = generated.cpu().detach().squeeze(0)
        generated = transforms.ToPILImage()(generated)
        return generated

    def save_result(self, image, save_path):
        """保存风格迁移结果"""
        if image:
            image.save(save_path)
            print(f'风格迁移结果已保存到: {save_path}')

def apply_style_transfer(content_path, style_path, output_path):
    """应用风格迁移的便捷函数"""
    if not TORCH_AVAILABLE:
        print("PyTorch未安装，无法使用风格迁移功能")
        return False

    try:
        style_transfer = StyleTransfer()
        result = style_transfer.transfer_style(content_path, style_path)
        if result:
            style_transfer.save_result(result, output_path)
            return True
        else:
            return False
    except Exception as e:
        print(f'风格迁移失败: {e}')
        return False

if __name__ == '__main__':
    # 示例用法
    content_path = 'test_content.jpg'
    style_path = 'test_style.jpg'
    output_path = 'output_style_transfer.jpg'

    if os.path.exists(content_path) and os.path.exists(style_path):
        success = apply_style_transfer(content_path, style_path, output_path)
        if success:
            print('风格迁移完成！')
        else:
            print('风格迁移失败')
    else:
        print('测试图像文件不存在')
        print('请准备content.jpg和style.jpg文件进行测试')
