import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint


class GestureControl:
    """基于肤色检测+轮廓分析的手势控制器"""

    def __init__(self):
        self._last_gesture = None
        self._gesture_count = 0
        self._min_gesture_count = 5

        self._last_distance = 0
        self._zoom_factor = 1.0
        self._last_hand_center = None
        self._pan_offset = QPoint(0, 0)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ── 肤色检测 ─────────────────────────────────────────────

    def _skin_mask(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        skin = ((cr >= 133) & (cr <= 183) & (cb >= 77) & (cb <= 133)).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, self._kernel, iterations=1)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        return skin

    # ── 手部区域检测 ─────────────────────────────────────────

    def detect_hand(self, frame):
        """基于肤色+轮廓检测手部区域"""
        mask = self._skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / max(w, 1)

            # 手部区域：高度/宽度在合理范围
            if 0.6 < aspect < 3.5 and w > 20 and h > 20:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    # 手部实体度通常 0.4-0.9
                    if 0.25 < solidity < 0.95:
                        hands.append((x, y, w, h, contour, hull))

        return hands

    # ── 手势识别 ─────────────────────────────────────────────

    def detect_gesture(self, frame):
        """检测手势，返回 (gesture_name, hand_center) 或 None"""
        hands = self.detect_hand(frame)

        if len(hands) == 0:
            self._last_gesture = None
            self._gesture_count = 0
            return None

        # 取最大的手部区域
        hands.sort(key=lambda h: h[2] * h[3], reverse=True)
        x, y, w_box, h_box, contour, hull = hands[0]
        hand_center = (x + w_box // 2, y + h_box // 2)

        # 凸缺陷手指计数
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                fingers = 0
                if defects is not None:
                    for d in defects:
                        s, e, f_idx, depth = d[0]
                        if depth > w_box * 0.12:
                            fingers += 1
                    fingers = min(fingers, 5)
            else:
                fingers = 0
        except Exception:
            fingers = 0

        # 根据手指数量判断手势
        if fingers == 0:
            gesture = 'fist'
        elif fingers == 1:
            gesture = 'one_finger'
        elif fingers == 2:
            gesture = 'two_fingers'
        else:
            gesture = 'open_hand'

        # 抗抖动：连续 N 帧一致才确认；手势变化时重置距离/位置
        if gesture == self._last_gesture:
            self._gesture_count += 1
        else:
            self._last_gesture = gesture
            self._gesture_count = 1
            self._last_distance = 0
            self._last_hand_center = None

        if self._gesture_count >= self._min_gesture_count:
            return gesture, hand_center
        else:
            return None, hand_center

    # ── 手势处理 ─────────────────────────────────────────────

    def handle_gesture(self, gesture, hand_center, photo_display=None):
        """将手势映射为控制动作"""
        if gesture == 'open_hand':
            return 'next'
        elif gesture == 'fist':
            return 'previous'
        elif gesture == 'two_fingers':
            if self._last_hand_center is not None:
                dx = hand_center[0] - self._last_hand_center[0]
                dy = hand_center[1] - self._last_hand_center[1]
                if dy > 15:
                    self._zoom_factor = min(3.0, self._zoom_factor * 1.08)
                    return 'zoom', self._zoom_factor
                elif dy < -15:
                    self._zoom_factor = max(0.3, self._zoom_factor * 0.92)
                    return 'zoom', self._zoom_factor
            self._last_hand_center = hand_center

        elif gesture == 'one_finger':
            if self._last_hand_center is not None:
                dx = hand_center[0] - self._last_hand_center[0]
                dy = hand_center[1] - self._last_hand_center[1]
                if abs(dx) > 5 or abs(dy) > 5:
                    self._pan_offset = QPoint(
                        self._pan_offset.x() + dx // 2,
                        self._pan_offset.y() + dy // 2
                    )
                    return 'pan', self._pan_offset
            self._last_hand_center = hand_center

        return None

    # ── 重置 ─────────────────────────────────────────────────

    def reset(self):
        self._last_gesture = None
        self._gesture_count = 0
        self._last_distance = 0
        self._zoom_factor = 1.0
        self._last_hand_center = None
        self._pan_offset = QPoint(0, 0)
