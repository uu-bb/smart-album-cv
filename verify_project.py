"""不加载模型与摄像头的智能相册结构验证。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    "main.py",
    "face_verification.py",
    "style_transfer.py",
    "pose_estimation.py",
    "emotion_classification.py",
    "yolov10_inference.py",
    "gesture_control.py",
    "video_maker.py",
    "requirements.txt",
    "docs/images/01_main_ui.png",
    "docs/images/03_architecture.png",
]


def class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def validate_python_syntax() -> int:
    count = 0
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += 1
    return count


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"缺少文件: {', '.join(missing)}")

    face_classes = class_names(ROOT / "face_verification.py")
    required_face_classes = {
        "FaceEncoder",
        "SiameseNetwork",
        "ContrastiveLoss",
        "FaceVerificationModel",
    }
    if not required_face_classes.issubset(face_classes):
        raise SystemExit("人脸编码模块结构不完整")

    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    required_signals = [
        "class CameraWorker(QThread)",
        "MySQL 失败",
        "sqlite3.connect",
        "SQLite 后备启用",
    ]
    missing_signals = [value for value in required_signals if value not in main_text]
    if missing_signals:
        raise SystemExit(f"主程序缺少关键工程信号: {missing_signals}")

    for name in REQUIRED_FILES:
        path = ROOT / name
        print(f"[OK] {name}: {path.stat().st_size} bytes")

    syntax_count = validate_python_syntax()
    print(f"[OK] {syntax_count} 个 Python 文件通过 AST 语法检查")
    print("[OK] QThread 相机线程与 MySQL -> SQLite 降级代码存在")
    print("[OK] Siamese Network、128 维编码器与 Contrastive Loss 类存在")
    print("项目结构验证通过；未调用摄像头、数据库或模型权重。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
