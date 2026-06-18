import os
import cv2
import numpy as np
import warnings

# 检查PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PoseEstimation:
    """人体姿态估计与腿部拉长"""

    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }

    POSE_PAIRS = [
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
        ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
        ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
        ["Nose", "LEye"], ["LEye", "LEar"]
    ]

    def __init__(self):
        self.net = None
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1

        if not TORCH_AVAILABLE:
            return

        model_pb = 'models/pose_deploy_linevec_faster_4_stages.pb'
        model_pbtxt = 'models/pose_deploy_linevec_faster_4_stages.pbtxt'

        if os.path.exists(model_pb) and os.path.exists(model_pbtxt):
            try:
                self.net = cv2.dnn.readNetFromTensorflow(model_pb, model_pbtxt)
            except Exception as e:
                warnings.warn(f"姿态估计模型加载失败: {e}，将使用简化方案")

    # ── 姿态估计 ──────────────────────────────────────────────

    def estimate_pose(self, image_path):
        """估计人体姿态，返回 (image, points)"""
        if not TORCH_AVAILABLE:
            return None, None

        image = cv2.imread(image_path)
        if image is None:
            return None, None

        h, w = image.shape[:2]

        if self.net is None:
            # 简化方案：基于默认人体比例返回估算关键点
            points = [None] * len(self.BODY_PARTS)
            points[self.BODY_PARTS["RHip"]] = (int(w * 0.47), int(h * 0.55))
            points[self.BODY_PARTS["LHip"]] = (int(w * 0.53), int(h * 0.55))
            points[self.BODY_PARTS["RKnee"]] = (int(w * 0.46), int(h * 0.75))
            points[self.BODY_PARTS["LKnee"]] = (int(w * 0.54), int(h * 0.75))
            points[self.BODY_PARTS["RAnkle"]] = (int(w * 0.45), int(h * 0.95))
            points[self.BODY_PARTS["LAnkle"]] = (int(w * 0.55), int(h * 0.95))
            return image, points

        try:
            inp_blob = cv2.dnn.blobFromImage(
                image, 1.0 / 255, (self.inWidth, self.inHeight),
                (0, 0, 0), swapRB=False, crop=False)
            self.net.setInput(inp_blob)
            output = self.net.forward()

            points = []
            for i in range(len(self.BODY_PARTS)):
                heatmap = output[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatmap)
                if conf > self.threshold:
                    x = int(w * point[0] / output.shape[3])
                    y = int(h * point[1] / output.shape[2])
                    points.append((x, y))
                else:
                    points.append(None)

            return image, points
        except Exception as e:
            print(f"姿态估计失败: {e}")
            return None, None

    # ── 绘制姿态 ──────────────────────────────────────────────

    def draw_pose(self, image, points):
        if image is None or points is None:
            return None
        out = image.copy()
        for i, pt in enumerate(points):
            if pt:
                cv2.circle(out, pt, 4, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(out, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        for pair in self.POSE_PAIRS:
            id_a = self.BODY_PARTS[pair[0]]
            id_b = self.BODY_PARTS[pair[1]]
            if points[id_a] and points[id_b]:
                cv2.line(out, points[id_a], points[id_b], (255, 0, 0), 2, cv2.LINE_AA)
        return out

    # ── 腿部拉长（分段式）─────────────────────────────────────

    def leg_lengthening(self, image, points, factor=1.15):
        """分段腿部拉长：大腿和小腿按不同比例拉伸，保持自然比例"""
        if image is None or points is None:
            return None

        rh = points[self.BODY_PARTS["RHip"]]
        rk = points[self.BODY_PARTS["RKnee"]]
        ra = points[self.BODY_PARTS["RAnkle"]]
        lh = points[self.BODY_PARTS["LHip"]]
        lk = points[self.BODY_PARTS["LKnee"]]
        la = points[self.BODY_PARTS["LAnkle"]]

        leg_keypoints = [rh, rk, ra, lh, lk, la]
        if not all(leg_keypoints):
            return None

        h, w = image.shape[:2]
        result = image.copy()

        hip_y = min(rh[1], lh[1])
        knee_y = int((rk[1] + lk[1]) / 2)
        ankle_y = max(ra[1], la[1])

        # 大腿：拉伸 70% 的 factor（保持比例）
        thigh_factor = 1.0 + (factor - 1.0) * 0.6
        # 小腿：拉伸 100% 的 factor（小腿拉长效果更明显）
        calf_factor = 1.0 + (factor - 1.0) * 1.0

        upper = result[:hip_y, :]
        thigh_region = result[hip_y:knee_y, :]
        calf_region = result[knee_y:ankle_y, :]
        lower = result[ankle_y:, :]

        if thigh_region.size > 0:
            new_thigh_h = int(thigh_region.shape[0] * thigh_factor)
            thigh_region = cv2.resize(thigh_region, (w, new_thigh_h), interpolation=cv2.INTER_CUBIC)

        if calf_region.size > 0:
            new_calf_h = int(calf_region.shape[0] * calf_factor)
            calf_region = cv2.resize(calf_region, (w, new_calf_h), interpolation=cv2.INTER_CUBIC)

        result = np.vstack([upper, thigh_region, calf_region, lower])

        # 缩回原始高度
        if result.shape[0] != h:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return result

    # ── 处理入口 ──────────────────────────────────────────────

    def process_image(self, image_path, output_path, factor=1.15):
        image, points = self.estimate_pose(image_path)
        if image is None:
            return False

        result = self.leg_lengthening(image, points, factor)
        if result is None:
            return False

        cv2.imencode('.jpg', result)[1].tofile(output_path)
        print(f"腿部拉长结果已保存: {output_path}")
        return True


def apply_pose_estimation(image_path, output_path, factor=1.15):
    """便捷入口"""
    try:
        estimator = PoseEstimation()
        return estimator.process_image(image_path, output_path, factor)
    except Exception as e:
        print(f"姿态估计失败: {e}")
        return False


if __name__ == '__main__':
    image_path = 'test_person.jpg'
    output_path = 'output_pose_estimation.jpg'
    if os.path.exists(image_path):
        success = apply_pose_estimation(image_path, output_path)
        print("完成" if success else "失败")
    else:
        print("请准备 test_person.jpg 进行测试")
