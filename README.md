# Smart Album CV

基于 PyQt5、OpenCV 与 PyTorch 的桌面智能相册原型：相机采集、相册管理、视觉算法集成与本地数据降级。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

- 无 MySQL 时自动降级为项目目录下的 SQLite；摄像头、模型文件缺失时对应 AI 功能不可用
- MySQL 配置参考 `.env.example`（通过环境变量设置，不要提交密码）

## 模块

| 文件 | 作用 |
|---|---|
| `main.py` | 主界面、相机、相册与数据库降级 |
| `face_verification.py` | Siamese 人脸编码与对比 |
| `style_transfer.py` | VGG19 风格迁移 |
| `pose_estimation.py` | 姿态估计 |
| `emotion_classification.py` | 表情分类 |
| `yolov10_inference.py` | YOLO 推理接口 |
| `gesture_control.py` | 手势控制 |
| `video_maker.py` | 图片视频生成 |
| `leg_slim_engine.py` | 图像处理引擎 |
| `init_db.py` | 数据库初始化 |
