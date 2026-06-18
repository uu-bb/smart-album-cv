"""
瘦腿引擎 v5 — 大腿+小腿分区域压缩 + 自适应腿宽渐变
"""
import cv2
import numpy as np
import time
import threading
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    mp = None

KP_SMOOTH_WINDOW = 8
TIME_BLEND_ALPHA = 0.80
KP_STABILITY_THRESH = 5.0
POSE_EVERY_N = 2


class LegSlimEngine:

    def __init__(self, model_path=None, lazy=True):
        self._lock = threading.Lock()
        self._kp_history = deque(maxlen=KP_SMOOTH_WINDOW)
        self._prev_result = None
        self._prev_map_x = None
        self._prev_map_y = None
        self._prev_kp_snapshot = None
        self._frame_counter = 0
        self._leg_type = "normal"
        self._stats = {"total_frames": 0, "total_time": 0.0}
        self._pose = None
        self._pose_ok = False

        if MEDIAPIPE_OK:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_ok = True
                print("[LegSlimEngine] MediaPipe Pose OK")
            except Exception as e:
                print(f"[LegSlimEngine] MediaPipe failed: {e}")

        self._photo_original = None
        self._undo_stack = deque(maxlen=8)

    # ── Public API ──────────────────────────────

    def process_frame(self, frame, strength, realtime=True):
        t0 = time.perf_counter()
        strength = float(max(0.0, min(1.0, strength)))
        result = self._realtime(frame, strength) if realtime else self._precise(frame, strength)
        self._stats["total_frames"] += 1
        self._stats["total_time"] += time.perf_counter() - t0
        return result

    def process_frame_with_history(self, frame, strength):
        self.push_undo_state(frame)
        return self._precise(frame, float(strength))

    def process_frame_precise(self, frame, strength):
        return self._precise(frame, float(strength))

    def push_undo_state(self, frame):
        self._undo_stack.append(frame.copy())
        self._photo_original = frame.copy()

    def undo_slim(self):
        return self._undo_stack.pop() if self._undo_stack else None

    def restore_original(self):
        return self._photo_original.copy() if self._photo_original is not None else None

    def leg_type_info(self):
        return {"type": "normal", "description": "标准型", "params": {}}

    def performance_stats(self):
        s = self._stats
        if s["total_frames"] == 0:
            return {}
        avg = s["total_time"] / s["total_frames"]
        return {"avg_ms": round(avg * 1000, 1), "fps_est": round(1.0 / avg, 1) if avg > 0 else 0}

    # ── Keypoint Detection ──────────────────────

    def _detect(self, frame):
        if not self._pose_ok:
            return None, 0.0
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        with self._lock:
            results = self._pose.process(rgb)
        rgb.flags.writeable = True
        if results.pose_landmarks is None:
            return None, 0.0

        lm = results.pose_landmarks.landmark
        px = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
        vis = lambda i: lm[i].visibility

        lh, rh = px(23), px(24)
        lk, rk = px(25), px(26)
        la, ra = px(27), px(28)

        conf = min(vis(23), vis(24), vis(25), vis(26), vis(27), vis(28))
        if conf < 0.3:
            return None, 0.0

        hip_spacing = abs(lh[0] - rh[0])
        leg_half_w = max(15, int(hip_spacing * 0.28))

        return {
            "lh": lh, "rh": rh, "lk": lk, "rk": rk, "la": la, "ra": ra,
            "hip_y": min(lh[1], rh[1]),
            "ankle_y": max(la[1], ra[1]),
            "knee_y": (lk[1] + rk[1]) // 2,
            "hip_spacing": hip_spacing,
            "leg_half_w": leg_half_w,
        }, conf

    def _smooth_kp(self, kp):
        """EMA smoothing with exponential weighting"""
        self._kp_history.append(kp)
        if len(self._kp_history) < 2:
            return kp

        alpha = 0.45
        result = {}
        keys_point = ["lh", "rh", "lk", "rk", "la", "ra"]
        keys_scalar = ["hip_y", "ankle_y", "knee_y", "hip_spacing", "leg_half_w"]

        for key in keys_scalar:
            values = [k[key] for k in self._kp_history if key in k]
            weights = [alpha * (1 - alpha) ** (len(values) - 1 - i) for i in range(len(values))]
            w_sum = sum(weights)
            result[key] = int(sum(v * w for v, w in zip(values, weights)) / w_sum)

        for key in keys_point:
            vals = [k[key] for k in self._kp_history if key in k]
            ws = [alpha * (1 - alpha) ** (len(vals) - 1 - i) for i in range(len(vals))]
            w_sum = sum(ws)
            result[key] = (int(sum(v[0] * w for v, w in zip(vals, ws)) / w_sum),
                           int(sum(v[1] * w for v, w in zip(vals, ws)) / w_sum))
        return result

    def _kp_shift(self, kp1, kp2):
        if kp1 is None or kp2 is None:
            return 999
        keys = ["lh", "rh", "lk", "rk", "la", "ra"]
        shifts = []
        for k in keys:
            if k in kp1 and k in kp2:
                shifts.append(abs(kp1[k][0] - kp2[k][0]) + abs(kp1[k][1] - kp2[k][1]))
        return sum(shifts) / len(shifts) if shifts else 999

    # ── Compression Profile (thigh vs calf) ─────

    def _comp(self, rel_y, strength):
        """Thigh: strong 0.30+0.40*s, Calf: moderate 0.40+0.35*s — higher strength = thinner"""
        if rel_y < 0.04:
            t = rel_y / 0.04
            return 1.0 - (1.0 - (0.30 + 0.40 * strength)) * t
        if 0.44 < rel_y < 0.56:
            knee_f = 1.0 - abs(rel_y - 0.50) * 10.0
            knee_f = max(0.15, knee_f)
            return 1.0 - (1.0 - (0.30 + 0.40 * strength)) * knee_f
        if rel_y > 0.92:
            fade = (1.0 - rel_y) / 0.08
            return 1.0 - (1.0 - (0.40 + 0.35 * strength)) * fade
        if rel_y < 0.45:
            return 0.30 + 0.40 * strength
        return 0.40 + 0.35 * strength

    def _leg_width(self, hip, knee, ankle, rel_y, half_w):
        """Natural leg width taper: hip=1.0 -> knee=0.90 -> ankle=0.35"""
        if rel_y < 0.45:
            factor = 1.0 - (rel_y / 0.45) * 0.10
        elif rel_y < 0.55:
            factor = 0.90 - ((rel_y - 0.45) / 0.10) * 0.20
        else:
            factor = 0.70 - ((rel_y - 0.55) / 0.45) * 0.35
        return max(3, int(half_w * factor))

    # ── Displacement Field ──────────────────────

    def _build_map(self, h, w, kp, strength):
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

        hip_y = kp["hip_y"]
        ankle_y = kp["ankle_y"]
        half_w = kp["leg_half_w"]

        for hip, knee, ankle in [(kp["lh"], kp["lk"], kp["la"]),
                                  (kp["rh"], kp["rk"], kp["ra"])]:
            for y in range(hip_y, min(ankle_y, h)):
                rel = (y - hip_y) / max(1, ankle_y - hip_y)
                comp = self._comp(rel, strength)
                if comp > 0.985:
                    continue

                cx = int(hip[0] + (ankle[0] - hip[0]) * rel)
                cur_hw = self._leg_width(hip, knee, ankle, rel, half_w)

                left = max(0, cx - cur_hw)
                right = min(w, cx + cur_hw)
                if right <= left + 2:
                    continue

                xs = np.arange(left, right, dtype=np.float32)
                src = cx + (xs - cx) / max(0.001, comp)
                src = np.clip(src, 0, w - 1)
                map_x[y, left:right] = src

        map_x = cv2.bilateralFilter(map_x, 5, 0.8, 3)
        map_x = cv2.GaussianBlur(map_x, (1, 5), 0.6)

        identity = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        top, bot = max(0, hip_y - 8), min(h, ankle_y + 8)
        map_x[:top] = identity[:top]
        map_x[bot:] = identity[bot:]

        return map_x.astype(np.float32), map_y.astype(np.float32)

    # ── Joint Protection ────────────────────────

    def _protect_joints(self, warped, original, kp, scale=1.0):
        h, w = warped.shape[:2]
        result = warped.copy()

        for y_center, h_ratio in [(kp["knee_y"], 0.018), (kp["ankle_y"], 0.010)]:
            ph = max(2, int(h * h_ratio))
            y0, y1 = max(0, int(y_center * scale) - ph), min(h, int(y_center * scale) + ph)
            for y in range(y0, y1):
                d = abs(y - int(y_center * scale)) / max(1, ph)
                alpha = np.clip(1.0 - d * 0.15, 0.85, 1.0)
                result[y] = (result[y].astype(np.float32) * alpha +
                             original[y].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return result

    # ── Realtime Mode ───────────────────────────

    def _realtime(self, frame, strength):
        self._frame_counter += 1
        h, w = frame.shape[:2]

        scale = 0.5
        sh, sw = int(h * scale), int(w * scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        kp = None
        if self._frame_counter % POSE_EVERY_N == 0:
            kp_raw, conf = self._detect(small)
            if kp_raw is not None:
                kp = self._smooth_kp(kp_raw)
        elif self._kp_history:
            kp = self._kp_history[-1]

        if kp is None:
            self._prev_result = None
            self._prev_map_x = None
            return frame
        if strength < 0.01:
            return frame

        shift = self._kp_shift(kp, self._prev_kp_snapshot)
        if shift < KP_STABILITY_THRESH and self._prev_map_x is not None:
            map_x = self._prev_map_x
            map_y = self._prev_map_y
        else:
            map_x, map_y = self._build_map(sh, sw, kp, strength)
            self._prev_map_x = map_x
            self._prev_map_y = map_y
            self._prev_kp_snapshot = kp.copy()

        warped = cv2.remap(small, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        warped = self._protect_joints(warped, small, kp, scale)

        result = cv2.resize(warped, (w, h), interpolation=cv2.INTER_LINEAR)

        if self._prev_result is not None:
            alpha = min(0.98, TIME_BLEND_ALPHA + strength * 0.10)
            result = cv2.addWeighted(result, alpha, self._prev_result, 1 - alpha, 0)
        self._prev_result = result.copy()

        return result

    # ── Photo Mode ──────────────────────────────

    def _precise(self, frame, strength):
        h, w = frame.shape[:2]

        max_dim = 1800
        if max(w, h) > max_dim:
            s = max_dim / max(w, h)
            work = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        else:
            work = frame.copy()
        wh, ww = work.shape[:2]

        kp_raw, conf = self._detect(work)
        if kp_raw is None or conf < 0.3:
            return frame

        kp = kp_raw

        map_x, map_y = self._build_map(wh, ww, kp, strength)
        map_x = cv2.GaussianBlur(map_x, (5, 1), 0.8)
        map_x = cv2.GaussianBlur(map_x, (1, 5), 0.7)

        result = cv2.remap(work, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
        result = self._protect_joints(result, work, kp)

        if max(w, h) > max_dim:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)

        self._album_result = result
        return result
