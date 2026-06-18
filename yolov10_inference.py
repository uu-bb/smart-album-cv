import os
import torch
import cv2
import numpy as np

# 检查 CUDA 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLOv10 模型定义（与训练脚本中相同）
class YOLOv10(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(YOLOv10, self).__init__()
        # 主干网络
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 512, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 512, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        # 颈部网络
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        # 头部网络
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes + 4)  # 4 个边界框坐标 + 2 个类别
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

# 加载模型
def load_yolov10_model(model_path):
    try:
        model = YOLOv10(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"加载YOLOv10模型失败: {e}")
        # 返回一个空模型
        return None

# 预处理图像
def preprocess_image(image, img_size=640):
    # 调整大小
    img = cv2.resize(image, (img_size, img_size))
    # 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为张量
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    # 添加批次维度
    img = img.unsqueeze(0)
    return img.to(device)

# 后处理输出
def postprocess_output(output, threshold=0.5):
    # 获取类别概率
    class_probs = torch.softmax(output[:, :2], dim=1)
    # 获取预测类别
    _, predicted_class = torch.max(class_probs, 1)
    # 获取边界框
    bbox = output[:, 2:].detach().cpu().numpy()[0]
    return predicted_class.item(), bbox

# 推理函数
def predict_with_yolov10(model, image):
    # 预处理图像
    input_tensor = preprocess_image(image)

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理输出
    predicted_class, bbox = postprocess_output(output)

    # 转换为标签
    labels = ['no_smile', 'smile']
    predicted_label = labels[predicted_class]

    return predicted_label, bbox

# 绘制结果
def draw_result(image, label, bbox):
    # 转换边界框坐标
    h, w = image.shape[:2]
    x, y, width, height = bbox
    x1 = int((x - width/2) * w)
    y1 = int((y - height/2) * h)
    x2 = int((x + width/2) * w)
    y2 = int((y + height/2) * h)

    # 确保坐标有效
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制标签
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

if __name__ == '__main__':
    # 加载模型
    model_path = 'models/yolov10_face.pth'
    if os.path.exists(model_path):
        model = load_yolov10_model(model_path)
        print('YOLOv10 模型加载成功')
    else:
        print(f'模型文件不存在: {model_path}')
        exit()

    # 测试图像
    test_image_path = 'data/smile_dataset/smile/16bc7269a9658bc8c65d17da68549919.jpg'
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)

        # 预测
        label, bbox = predict_with_yolov10(model, image)
        print(f'预测结果: {label}')
        print(f'边界框: {bbox}')

        # 绘制结果
        result_image = draw_result(image, label, bbox)

        # 保存结果
        output_path = 'yolov10_result.jpg'
        cv2.imwrite(output_path, result_image)
        print(f'结果已保存到 {output_path}')
    else:
        print(f'测试图像不存在: {test_image_path}')
