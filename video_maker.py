"""
多图自动剪辑生成视频 — Ken Burns 效果 + 交叉淡入淡出
"""
import cv2
import numpy as np
import os
import time


def _ken_burns_frame(img, progress, target_w, target_h):
    """Ken Burns 效果：缓慢平移 + 缩放"""
    h, w = img.shape[:2]
    scale_start = 1.0
    scale_end = 1.12
    pan_x_start, pan_y_start = 0.0, 0.0
    pan_x_end = (np.random.random() - 0.5) * 0.15
    pan_y_end = (np.random.random() - 0.5) * 0.15

    scale = scale_start + (scale_end - scale_start) * progress
    pan_x = pan_x_start + (pan_x_end - pan_x_start) * progress
    pan_y = pan_y_start + (pan_y_end - pan_y_start) * progress

    crop_w = int(target_w / scale)
    crop_h = int(target_h / scale)
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    cx = w // 2 + int(pan_x * w)
    cy = h // 2 + int(pan_y * h)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def make_slideshow(photo_paths, output_path='output.mp4', duration_per_photo=3.0,
                   transition_duration=0.8, fps=30, resolution=(1920, 1080),
                   progress_callback=None):
    """
    将多张照片生成 Ken Burns 风格幻灯片视频。

    Args:
        photo_paths: 照片路径列表
        output_path: 输出视频路径
        duration_per_photo: 每张照片显示时长（秒）
        transition_duration: 过渡时长（秒）
        fps: 帧率
        resolution: (宽, 高)
        progress_callback: 可选回调(total, current)
    """
    if len(photo_paths) < 2:
        return False

    target_w, target_h = resolution
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    frames_per_photo = int(duration_per_photo * fps)
    trans_frames = int(transition_duration * fps)
    total_frames = len(photo_paths) * frames_per_photo - (len(photo_paths) - 1) * (frames_per_photo - trans_frames)

    prev_frame = None
    frame_idx = 0

    for pi, path in enumerate(photo_paths):
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue

        for fi in range(frames_per_photo):
            progress = fi / max(1, frames_per_photo - 1)
            current = _ken_burns_frame(img, progress, target_w, target_h)

            if fi < trans_frames and prev_frame is not None:
                alpha = fi / trans_frames
                current = cv2.addWeighted(current, alpha, prev_frame, 1.0 - alpha, 0)

            out.write(current)
            frame_idx += 1
            if progress_callback:
                progress_callback(total_frames, frame_idx)

        prev_frame = _ken_burns_frame(img, 1.0, target_w, target_h)

    out.release()
    return True


def create_text_overlay(img, text, position='bottom'):
    """在图片上叠加文字"""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = min(w, h) / 800.0
    thickness = max(1, int(font_scale * 2))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if position == 'bottom':
        x = (w - text_size[0]) // 2
        y = h - 30

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 20, y - text_size[1] - 20), (x + text_size[0] + 20, y + 20), (0, 0, 0, 128), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return img
