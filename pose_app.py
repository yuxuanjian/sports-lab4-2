import sys, os, time, csv
import numpy as np
import cv2
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

# ==============================
# ====== 新增：角度計算工具 ======
# ==============================
def calculate_angle(a, b, c):
    """
    計算三點間的夾角 (在 B 點的夾角)
    輸入為 (x, y) 坐標
    """
    # 轉換為 numpy 陣列
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    # 檢查是否有 NaN (缺失點)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan
    
    # 計算向量 BA 和 BC
    ba = a - b
    bc = c - b
    
    # 計算點積 (dot product) 和模長 (magnitude)
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # 避免除以零
    if norm_ba == 0 or norm_bc == 0:
        return np.nan
        
    # 計算夾角的餘弦值
    cosine_angle = dot_product / (norm_ba * norm_bc)
    
    # 限制數值在 -1 到 1 之間 (避免浮點數誤差)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # 計算角度 (弧度) 並轉換為度數
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# ==============================
# 常數（寫死設定）
# ==============================
APP_TITLE = "Pose App - MediaPipe / YOLOv8 (Lab4-2)"
FIXED_W, FIXED_H = 960, 540          # 擷取解析度（推論/CSV 皆用這個）
MP_MODEL_COMPLEXITY = 1              # MediaPipe 複雜度：0/1/2
YOLO_WEIGHTS = Path(__file__).with_name("yolov8n-pose.pt") # YOLO 權重（同資料夾）

# ====== 數據後處理常數 ======
VISIBILITY_THRESHOLD = 0.5   # 關鍵點可見度/信心度門檻 (低於此值視為 NaN)
MAX_GAP_MS = 150             # 最大插補時間 (ms)，超過此時間的缺失點將維持 NaN
EMA_ALPHA = 0.2              # EMA 平滑因子 (0.0~1.0, 越小越平滑但延遲越高)

# ====== (修改) Lab4-2 平滑器選擇 ======
# 為了滿足 Lab4-2 Q2 報告要求，可在此切換平滑器
# 選項: "EMA" 或 "KALMAN"
SMOOTHER_TYPE = "KALMAN"
KALMAN_R = 0.1   # 測量雜訊 (R 越小，越相信測量值，越不平滑)
KALMAN_Q = 1e-4  # 過程雜訊 (Q 越小，越相信模型預測，越平滑)
# ==============================

# ==============================
# 視覺畫布：鋪滿顯示 + HUD（FPS）(不變)
# ==============================
class VideoCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self._fps_text = "FPS: -"
        self.setStyleSheet("background:#222;")
        self.setMinimumSize(1, 1)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    @QtCore.pyqtSlot(QtGui.QImage)
    def setFrame(self, img: QtGui.QImage):
        if img is None:
            self._pix = None
        else:
            self._pix = QtGui.QPixmap.fromImage(img)
        self.update()

    @QtCore.pyqtSlot(float)
    def setFps(self, fps: float):
        self._fps_text = f"FPS: {fps:.1f}"

    def reset(self):
        self._pix = None
        self._fps_text = "FPS: -"
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        # 背景
        p.fillRect(self.rect(), QtGui.QColor("#222"))

        # 影像：維持比例且鋪滿（必要時裁切），置中顯示
        if self._pix:
            target_size = self.size()
            src_size = self._pix.size()
            scaled_size = src_size.scaled(target_size, QtCore.Qt.KeepAspectRatioByExpanding)
            x = (target_size.width() - scaled_size.width()) // 2
            y = (target_size.height() - scaled_size.height()) // 2
            p.drawPixmap(QtCore.QRect(QtCore.QPoint(x, y), scaled_size), self._pix)
        else:
            # 初始/停止狀態提示
            p.setPen(QtGui.QColor(180, 180, 180))
            font = p.font(); font.setPointSize(16); p.setFont(font)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Stopped – press Start")

        # HUD：FPS（永遠在可視範圍）
        if self._fps_text:
            margin = 10
            font = p.font(); font.setPointSize(14); font.setBold(True); p.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            text_w = metrics.horizontalAdvance(self._fps_text)
            text_h = metrics.height()
            bg_rect = QtCore.QRect(margin-6, margin-6, text_w+12, text_h+12)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 140))
            p.drawRoundedRect(bg_rect, 6, 6)
            p.setPen(QtGui.QColor(0, 255, 0))
            p.drawText(margin, margin + text_h, self._fps_text)
        p.end()

# ==============================
# 後端共用介面 (不變)
# ==============================
class PoseBackend(QtCore.QObject):
    def start(self):...
    def stop(self):...
    def process(self, frame_bgr):...
    def header(self):...
    def name(self):...
    # (填補) 確保所有後端都有 names 屬性
    names = [] 

# ==============================
# MediaPipe 後端 (填補 names)
# ==============================
class MediaPipeBackend(PoseBackend):
    def __init__(self):
        super().__init__()
        self.mp = None
        self.pose = None
        self.drawing = None
        # (填補) MediaPipe 的 names 屬性會在 start() 裡被填充，這裡先給個空列表
        self.names = []

    def name(self): return "MediaPipe"

    def start(self):
        import mediapipe as mp
        self.mp = mp
        self.drawing = mp.solutions.drawing_utils
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # (*** Lab4-2 備註： MediaPipe (BlazePose) 有 33 個點 ***)
        self.names = [lm.name for lm in mp.solutions.pose.PoseLandmark]

    def stop(self):
        if self.pose:
            self.pose.close()
        self.pose = None

    def header(self):
        h = []
        for n in self.names:
            h += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_vis"]
        return h

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        row = []
        if res.pose_landmarks:
            self.drawing.draw_landmarks(
                frame_bgr, res.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
                self.drawing.DrawingSpec(thickness=2, circle_radius=2),
                self.drawing.DrawingSpec(thickness=2)
            )
            lms = res.pose_landmarks.landmark
            for i in range(len(self.names)):
                lm = lms[i]
                row += [lm.x, lm.y, lm.z, lm.visibility]
        else:
            for _ in range(len(self.names)):
                row += [np.nan, np.nan, np.nan, np.nan]
        return frame_bgr, row

# ==============================
# YOLOv8 後端 (填補 names)
# ==============================
class YOLOv8Backend(PoseBackend):
    def __init__(self):
        super().__init__()
        self.model = None
        # (*** Lab4-2 備註： YOLOv8 (COCO) 只有 17 個點 ***)
        self.names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def name(self): return "YOLOv8"

    def start(self):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("未安裝 ultralytics，請先執行：pip install ultralytics") from e
        model_path = str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n-pose.pt"
        self.model = YOLO(model_path)

    def stop(self):
        self.model = None

    def header(self):
        h = []
        for n in self.names:
            h += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_vis"]
        return h

    def process(self, frame_bgr):
        results = self.model(frame_bgr, verbose=False)
        r = results[0] # 取第一個結果
        annotated = r.plot()
        row = []
        if r.keypoints is not None and len(r.keypoints) > 0 and r.keypoints.xy.shape[0] > 0:
            # 假設我們只處理第一個偵測到的人
            kp_xy = r.keypoints.xy[0].cpu().numpy()      # (17, 2) 第一位人物
            kp_conf = None
            if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None and r.keypoints.conf.shape[0] > 0:
                kp_conf = r.keypoints.conf[0].cpu().numpy() # (17,)
                
            h, w = annotated.shape[:2]
            for i in range(len(self.names)):
                # 確保索引 i 在 kp_xy 範圍內
                if i < len(kp_xy):
                    x, y = kp_xy[i]
                    # 確保索引 i 在 kp_conf 範圍內
                    vis = float(kp_conf[i]) if kp_conf is not None and i < len(kp_conf) else np.nan
                    # YOLOv8 的 Z 軸為 NaN，以匹配 MediaPipe 的欄位
                    row += [float(x)/w, float(y)/h, np.nan, vis]
                else:
                    # 如果關鍵點數量少於預期 (e.g., 只有部分點被偵測到，通常不會發生，但以防萬一)
                    row += [np.nan, np.nan, np.nan, np.nan]
        else:
            for _ in range(len(self.names)):
                row += [np.nan, np.nan, np.nan, np.nan]
        return annotated, row

# ==============================
# ====== 平滑器 1：EMA ======
# ==============================
class EMA:
    """單一數值的指數移動平均平滑器 (Lab4-2 方法 1)"""
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.last_value = np.nan
        
    def update(self, new_value):
        """更新平滑器並返回平滑後的值"""
        if np.isnan(new_value):
            # 不應使用 NaN 來更新
            return self.last_value
            
        if np.isnan(self.last_value):
            # 這是第一個有效值
            self.last_value = new_value
        else:
            # EMA 公式
            self.last_value = self.alpha * new_value + (1 - self.alpha) * self.last_value
        return self.last_value
    
    def get_last(self):
        """獲取最後的平滑值 (用於插補)"""
        return self.last_value

# ==============================
# ====== (新增) 平滑器 2：Kalman ======
# ==============================
class Kalman1DSmoother:
    """
    實作一個簡易的 1D 卡爾曼濾波器 (Lab4-2 方法 2)
    狀態模型為 [position, velocity] (即 [angle, angular_rate])。
    符合 Lab4-2.pdf Page 6 的要求。
    """
    
    def __init__(self, R: float = 0.1, Q: float = 1e-4):
        self.R_val = R # 測量雜訊 (R 越小，越相信測量值)
        self.Q_val = Q # 過程雜訊 (Q 越小，越相信模型預測)
        
        self.x = np.array([0.0, 0.0]) # 狀態 [pos, vel]
        self.P = np.eye(2) * 1.0       # 狀態協方差
        
        # 狀態轉移矩陣 F (假設 dt=1 幀)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
                            
        # 過程雜訊協方差矩陣 Q
        # 根據 Lab4-2.pdf (Page 6)，Q 的對角線元素都使用 Q_val
        self.Q_matrix = np.array([[self.Q_val, 0.0],
                                  [0.0, self.Q_val]])
                                    
        # 觀測矩陣 H (我們只能觀測到 pos)
        self.H = np.array([[1.0, 0.0]])
        
        self.initialized = False

    def update(self, measurement):
        """使用新的測量值更新濾波器。"""
        
        measurement = float(measurement) # 確保為浮點數

        if np.isnan(measurement):
            # 如果沒有測量值 (NaN)，我們只能進行預測
            if self.initialized:
                self._predict()
                # 返回預測位置 (x[0])
                return self.x[0] 
            else:
                return np.nan
                
        if not self.initialized:
            # 使用第一個有效測量值來初始化狀態
            self.x[0] = measurement # 位置
            self.x[1] = 0.0         # 速度
            # P 矩陣維持初始設定
            self.initialized = True
            return measurement

        # --- 1. 預測 (Predict) ---
        self._predict()
        
        # --- 2. 更新 (Update) ---
        # 測量殘差 (y = z - Hx)
        y = measurement - (self.H @ self.x)[0] 
        # 殘差協方差 (S = H P H.T + R)
        # 這裡 R 是一個純量 self.R_val
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R_val 
        # 卡爾曼增益 (K = P H.T S^-1)
        K = self.P @ self.H.T * (1.0/S) 
        
        # 更新狀態估計 (x = x + K y)
        self.x = self.x + K.flatten() * y 
        # 更新狀態協方差 (P = (I - K H) P)
        self.P = (np.eye(2) - K @ self.H) @ self.P 
        
        # 回傳平滑後的位置 (角度)
        return self.x[0]

    def _predict(self):
        """ 內部的預測步驟 """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_matrix

    def get_last(self):
        """獲取最後的平滑值 (用於插補)"""
        # 返回位置 (x[0])
        return self.x[0] if self.initialized else np.nan


# ==============================
# ====== [版本 V3_Kalman：QC + Kalman] ======
# ==============================
class DataProcessor:
    """
    [版本 V3_Kalman：Kalman 濾波對照組]
    執行 QC (Gating + F-Fill) 和 Kalman 平滑。
    """
    def __init__(self, landmark_names):
        self.num_lms = len(landmark_names)
        self.num_values = self.num_lms * 4 
        
        # 1. (*** 強制使用 KALMAN ***)
        self.smoothers = [] 
        print(f"DataProcessor: [VERSION V3_KALMAN] 正在初始化 KALMAN 平滑器...")
        
        for i in range(self.num_lms):
            self.smoothers.append(Kalman1DSmoother(R=KALMAN_R, Q=KALMAN_Q)) # x
            self.smoothers.append(Kalman1DSmoother(R=KALMAN_R, Q=KALMAN_Q)) # y
            self.smoothers.append(Kalman1DSmoother(R=KALMAN_R, Q=KALMAN_Q)) # z
            self.smoothers.append(None) # vis (vis 不需要平滑)
            
        # 2. 插補的時間戳
        self.last_valid_ts = np.full(self.num_values, 0.0)
        
        # 3. FSM 狀態
        self.fsm_state = "DOWN"
        self.rep_count = 0
        self.shoulder_angle_thresholds = {"UP": 90, "DOWN": 90} 

        # 4. 關節點索引 (維持不變)
        def find_idx(names, target):
            target_upper = target.upper()
            target_lower = target.lower()
            for i, name in enumerate(names):
                name_str = str(name)
                if name_str == target_upper or name_str == target_lower:
                    return i
            return None 
            
        self.idx_r_shoulder = find_idx(landmark_names, "RIGHT_SHOULDER")
        self.idx_r_elbow = find_idx(landmark_names, "RIGHT_ELBOW")
        self.idx_r_wrist = find_idx(landmark_names, "RIGHT_WRIST")
        
        if self.idx_r_shoulder is None or self.idx_r_elbow is None or self.idx_r_wrist is None:
             print("警告： FSM 需要的 RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST 找不到。")


    def _get_point_xy(self, row_data, landmark_idx):
        if landmark_idx is None:
            return np.array([np.nan, np.nan])
        base_idx = landmark_idx * 4
        if base_idx + 1 >= len(row_data):
             return np.array([np.nan, np.nan])
        x = row_data[base_idx]
        y = row_data[base_idx + 1]
        return np.array([x, y])

    def _update_fsm_and_angles(self, row_data):
        """(FSM 核心) 根據處理後的數據更新 FSM 和角度"""
        
        # --- 1. 提取點 ---
        p_r_shoulder = self._get_point_xy(row_data, self.idx_r_shoulder)
        p_r_elbow = self._get_point_xy(row_data, self.idx_r_elbow)
        p_r_wrist = self._get_point_xy(row_data, self.idx_r_wrist)
        
        # --- 2. 計算角度 ---
        p_virtual_down = np.array([p_r_shoulder[0], p_r_shoulder[1] + 1.0])
        shoulder_vertical_angle = calculate_angle(p_r_elbow, p_r_shoulder, p_virtual_down)

        # --- 3. 更新 FSM ---
        r_wrist_y = p_r_wrist[1]
        r_shoulder_y = p_r_shoulder[1]
        
        fsm_values_valid = (
            not np.isnan(shoulder_vertical_angle) and
            not np.isnan(p_r_wrist).any() and
            not np.isnan(p_r_shoulder).any()
        )
        
        if fsm_values_valid:
            is_hand_high = r_wrist_y < r_shoulder_y 
            
            if self.fsm_state == "DOWN":
                if shoulder_vertical_angle > self.shoulder_angle_thresholds["UP"] and is_hand_high:
                    self.fsm_state = "UP"
                    self.rep_count += 1
            elif self.fsm_state == "UP":
                if shoulder_vertical_angle < self.shoulder_angle_thresholds["DOWN"]:
                    self.fsm_state = "DOWN"
                    
        
        stats = {
            "reps": self.rep_count,
            "fsm_state": self.fsm_state,
            "shoulder_v_angle": shoulder_vertical_angle, 
            "r_wrist_y": r_wrist_y,                
            "r_shoulder_y": r_shoulder_y,          
            "shoulder_pos_xy": p_r_shoulder        
        }
        return stats

    def process(self, elapsed_ms, row_data):
        """(process 核心邏輯保持不變)"""
        processed_row = []
        
        if len(row_data)!= self.num_values:
            nan_row = [np.nan] * self.num_values
            stats = self._update_fsm_and_angles(nan_row)
            return nan_row, stats

        for i in range(self.num_lms):
            idx_base = i * 4
            x, y, z, vis = row_data[idx_base : idx_base + 4]
            
            is_gated = (vis < VISIBILITY_THRESHOLD) or np.isnan(vis)
            coords_in = [x, y, z]
            coords_out = []
            
            for j in range(3): 
                coord_idx = idx_base + j
                val = coords_in[j]
                
                if is_gated or np.isnan(val):
                    if (elapsed_ms - self.last_valid_ts[coord_idx]) <= MAX_GAP_MS:
                        # 這裡將使用 Kalman.get_last()
                        coords_out.append(self.smoothers[coord_idx].get_last())
                    else:
                        coords_out.append(np.nan)
                else:
                    self.last_valid_ts[coord_idx] = elapsed_ms
                    # 這裡將使用 Kalman.update()
                    smoothed_val = self.smoothers[coord_idx].update(val)
                    coords_out.append(smoothed_val)

            processed_vis = vis if not is_gated else np.nan
            processed_row.extend(coords_out)
            processed_row.append(processed_vis)
        
        stats = self._update_fsm_and_angles(processed_row)
        return processed_row, stats

# ==============================
# 擷取執行緒（背景執行） (*** 已修改 ***)
# ==============================
class CaptureWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage, float)  # (frame, fps)
    rowReady = QtCore.pyqtSignal(list)                   # CSV row（含 frame_id/timestamp）

    def __init__(self, cam_index, backend: PoseBackend, parent=None):
        super().__init__(parent)
        self.cam_index = int(cam_index)
        self.backend = backend # backend 實例 (已在 MainWindow 被 start/stop 過)
        self._running = False
        self._frame_id = 0
        self._fps_interval = 0.5
        self._last_update = time.time()
        self._cnt = 0
        self._fps = 0.0
        self.t0 = None # 起算時間（perf_counter）

        # ====== (修改) 初始化數據處理器 ======
        # backend.names 此刻應該已經被 MainWindow 的 _prepare_backend 填充
        self.processor = DataProcessor(landmark_names=self.backend.names)
        # ==================================

    def run(self):
        # Windows：DSHOW；跨平台可移除第二參數
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FIXED_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FIXED_H)
        if not cap.isOpened():
            QtCore.qWarning("Cannot open camera")
            return

        self.backend.start()
        self._running = True
        self._frame_id = 0
        self._cnt = 0
        self._fps = 0.0
        self.t0 = time.perf_counter() # 相對時間起點
        
        # 重設 processor 狀態
        self.processor = DataProcessor(landmark_names=self.backend.names)
        self._last_update = time.time()


        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (FIXED_W, FIXED_H), interpolation=cv2.INTER_LINEAR)

                self._frame_id += 1
                ts_ms = int(time.time() * 1000)
                elapsed_ms = int((time.perf_counter() - self.t0) * 1000)

                # 1. 後端處理 (原始數據)
                frame, row_data_raw = self.backend.process(frame)
                
                # ====== (修改) 進行數據後處理 ======
                # 現在 process 返回 (processed_row, stats)
                row_data_processed, stats = self.processor.process(elapsed_ms, row_data_raw)
                # ==================================

                # ====== (*** 已修改 ***) 將統計數據繪製到影像上 ======
                reps = stats["reps"]
                state = stats["fsm_state"]
                
                # (+) 新增 FSM 除錯數據
                shoulder_angle_fsm = stats["shoulder_v_angle"]
                wrist_y = stats["r_wrist_y"]
                shoulder_y = stats["r_shoulder_y"]
                
                # ====== vvv (*** 新增這行 ***) vvv ======
                shoulder_pos_xy = stats["shoulder_pos_xy"] # 接收肩膀的 X,Y 坐標
                # ====== ^^^ (*** 新增這行 ***) ^^^ ======

                
                # --- vvv [修改] 擴充右上角的資訊框 vvv ---
                
                # 定義新尺寸
                BOX_W = 250  # 矩形寬度
                BOX_H = 170  # (*** 您的新高度 ***)
                PADDING = 15 # 文字邊距
                
                # 右上角矩形
                cv2.rectangle(frame, (FIXED_W - BOX_W, 0), (FIXED_W, BOX_H), (50, 50, 50), -1)
                
                # (*** 您的新 Y 座標和字體 ***)
                LINE_H = 22  # 每行高度
                START_Y = 70 # 起始 Y 座標
                FONT_SCALE = 0.6
                FONT_SCALE_SMALL = 0.4
                
                # REPS 文字
                cv2.putText(frame, f"REPS: {reps}", 
                            (FIXED_W - BOX_W + PADDING, START_Y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE, (0, 255, 0), 1, cv2.LINE_AA)
                
                # STATE 文字
                state_color = (0, 255, 0) if state == "UP" else (0, 180, 255)
                cv2.putText(frame, f"STATE: {state}", 
                            (FIXED_W - BOX_W + PADDING, START_Y + LINE_H), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE, state_color, 1, cv2.LINE_AA)

                # 肩膀角度 (FSM)
                angle_str = f"{shoulder_angle_fsm:.1f}" if not np.isnan(shoulder_angle_fsm) else "NaN"
                cv2.putText(frame, f"S-Angle: {angle_str} deg", 
                            (FIXED_W - BOX_W + PADDING, START_Y + LINE_H*2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE_SMALL, (255, 255, 0), 1, cv2.LINE_AA) # 藍色

                # 肩膀 Y 座標
                s_y_str = f"{shoulder_y:.3f}" if not np.isnan(shoulder_y) else "NaN"
                cv2.putText(frame, f"Shoulder Y: {s_y_str}", 
                            (FIXED_W - BOX_W + PADDING, START_Y + LINE_H*3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE_SMALL, (255, 255, 255), 1, cv2.LINE_AA) # 白色

                # 手腕 Y 座標
                w_y_str = f"{wrist_y:.3f}" if not np.isnan(wrist_y) else "NaN"
                cv2.putText(frame, f"Wrist Y: {w_y_str}", 
                            (FIXED_W - BOX_W + PADDING, START_Y + LINE_H*4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            FONT_SCALE_SMALL, (255, 255, 255), 1, cv2.LINE_AA) # 白色

                # --- ^^^ [修改] 繪製計數和狀態 (結束) ^^^ ---
                
                
                # ====== vvv (*** 新增區塊 ***) vvv ======
                # 在右肩膀旁繪製 "S-Angle" (垂直肩部) 角度
                
                # 檢查角度和位置是否有效 (非 NaN)
                if not np.isnan(shoulder_angle_fsm) and not np.isnan(shoulder_pos_xy).any():
                    # 將正規化坐標 (0-1) 轉換為像素坐標
                    x = int(shoulder_pos_xy[0] * FIXED_W) 
                    y = int(shoulder_pos_xy[1] * FIXED_H) 
                    
                    # 在右肩關節旁顯示角度 (使用紫色)
                    cv2.putText(frame, f"{shoulder_angle_fsm:.0f}", 
                                (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 0, 255), 2, cv2.LINE_AA)
                # ====== ^^^ (*** 新增區塊 ***) ^^^ ======
                
                
                # ==================================

                # FPS 計算
                self._cnt += 1
                now = time.time()
                if (now - self._last_update) >= self._fps_interval:
                    delta = now - self._last_update
                    self._fps = self._cnt / delta if delta > 0 else 0.0
                    self._cnt = 0
                    self._last_update = now

                # CSV (使用處理後的數據)
                row = [self._frame_id, ts_ms, elapsed_ms] + row_data_processed
                self.rowReady.emit(row)

                # 送影像到 GUI
                qimg = QtGui.QImage(frame.data, FIXED_W, FIXED_H,
                                    frame.strides[0], QtGui.QImage.Format_BGR888)
                self.frameReady.emit(qimg.copy(), self._fps)

                QtCore.QThread.msleep(1)
        finally:
            cap.release()
            self.backend.stop()

    def stop(self):
        self._running = False
        self.wait(1000)

# ==============================
# 主視窗 (填補 combobox 內容)
# ==============================
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)

        # 控制列
        # (填補) combobox 內容
        self.cmb_backend = QtWidgets.QComboBox()
        self.cmb_backend.addItems(["MediaPipe", "YOLOv8"]) 
        self.spn_cam = QtWidgets.QSpinBox(); self.spn_cam.setRange(0, 16); self.spn_cam.setValue(0)
        self.edt_csv = QtWidgets.QLineEdit("teamXX_output.csv")
        self.btn_browse = QtWidgets.QPushButton("…")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.lbl_top_fps = QtWidgets.QLabel("FPS: -")
        self.lbl_elapsed = QtWidgets.QLabel("Elapsed: 0 s")    # <<< 新增：經過秒數顯示（整秒）

        # 影像畫布
        self.canvas = VideoCanvas()

        # 版面
        form = QtWidgets.QFormLayout()
        form.addRow("Backend", self.cmb_backend)
        form.addRow("Camera Index", self.spn_cam)
        csvrow = QtWidgets.QHBoxLayout()
        csvrow.addWidget(self.edt_csv); csvrow.addWidget(self.btn_browse)
        form.addRow("CSV Output", csvrow)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.btn_start); btnrow.addWidget(self.btn_stop)
        btnrow.addStretch(1)
        btnrow.addWidget(self.lbl_elapsed)  # 放在右側
        btnrow.addSpacing(12)
        btnrow.addWidget(self.lbl_top_fps)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(form)
        layout.addLayout(btnrow)
        layout.addWidget(self.canvas, 1)

        # 狀態
        self.worker = None
        self.csv_file = None
        self.csv_writer = None
        self.accept_frames = False # 幀過濾旗標

        # 計時器（以秒為單位）
        self._ui_t0 = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200) # 200ms 更新一次，但顯示整秒
        self._timer.timeout.connect(self._tick_elapsed)

        # 事件
        self.btn_browse.clicked.connect(self.onBrowse)
        self.btn_start.clicked.connect(self.onStart)
        self.btn_stop.clicked.connect(self.onStop)
        self.cmb_backend.currentTextChanged.connect(self.onBackendChange)

        self.onBackendChange(self.cmb_backend.currentText())

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key_B and self.btn_stop.isEnabled():
            self.toggleBackendInRuntime()

    def onBackendChange(self, text):
        self.edt_csv.setText("teamXX_output_mediapipe.csv" if text == "MediaPipe" else "teamXX_output_yolo.csv")

    def onBrowse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose CSV", self.edt_csv.text(), "CSV (*.csv)")
        if path: self.edt_csv.setText(path)

    def makeBackend(self) -> PoseBackend:
        return MediaPipeBackend() if self.cmb_backend.currentText() == "MediaPipe" else YOLOv8Backend()

    def openCsv(self, kp_header):
        path = self.edt_csv.text().strip()
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        try:
            self.csv_file = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            header = ["frame", "timestamp_ms", "elapsed_ms"] + kp_header
            self.csv_writer.writerow(header)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "CSV 開啟失敗", f"無法寫入檔案：{path}\n錯誤：{e}")
            self.csv_file = None
            self.csv_writer = None

    def closeCsv(self):
        if self.csv_file:
            try: self.csv_file.flush()
            except: pass
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    def _start_elapsed_timer(self):
        self._ui_t0 = time.perf_counter()
        self.lbl_elapsed.setText("Elapsed: 0 s")
        self._timer.start()

    def _stop_elapsed_timer(self):
        self._timer.stop()
        self._ui_t0 = None
        self.lbl_elapsed.setText("Elapsed: 0 s")

    def _tick_elapsed(self):
        if self._ui_t0 is None:
            return
        secs = int(time.perf_counter() - self._ui_t0) # 整秒
        self.lbl_elapsed.setText(f"Elapsed: {secs} s")

    def _prepare_backend(self):
        """
        (修改) 嘗試啟動後端並獲取其標頭，用於初始化 worker
        """
        try:
            backend = self.makeBackend()
            backend.start()  # 啟動以填充.names
            kp_header = backend.header()
            backend.stop()   # 立即停止，worker 將重新啟動
            return backend, kp_header
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "後端啟動失敗", str(e))
            return None, None

    def onStart(self):
        if self.worker and self.worker.isRunning(): return
        
        # 1. 準備後端並獲取標頭
        backend_instance, kp_header = self._prepare_backend()
        if backend_instance is None:
            return

        # 2. 開啟 CSV
        self.openCsv(kp_header)
        if self.csv_writer is None:
            return # 開啟 CSV 失敗

        # 3. 建立 Worker (傳入已初始化的後端實例)
        self.worker = CaptureWorker(
            cam_index=int(self.spn_cam.value()),
            backend=backend_instance
        )

        # 4. 連線
        self.worker.frameReady.connect(self.onFrame)
        self.worker.rowReady.connect(self.onRow)
        self.accept_frames = True
        self.worker.start()

        # 5. 啟動計時
        self._start_elapsed_timer()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def onStop(self):
        self.accept_frames = False
        if self.worker:
            try:
                self.worker.frameReady.disconnect(self.onFrame)
                self.worker.rowReady.disconnect(self.onRow)
            except TypeError:
                pass
            self.worker.stop()
            self.worker = None

        self.closeCsv()
        self.canvas.reset()
        self.lbl_top_fps.setText("FPS: -")
        self._stop_elapsed_timer()

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def toggleBackendInRuntime(self):
        if not self.worker: return

        # 1. 停止舊的
        self.accept_frames = False
        try:
            self.worker.frameReady.disconnect(self.onFrame)
            self.worker.rowReady.disconnect(self.onRow)
        except TypeError:
            pass
        self.worker.stop()
        self.worker = None
        self.closeCsv()
        self.canvas.reset()
        self.lbl_top_fps.setText("FPS: -")
        self._stop_elapsed_timer()

        # 2. 切換後端
        current_backend = self.cmb_backend.currentText()
        new_backend_name = "YOLOv8" if current_backend == "MediaPipe" else "MediaPipe"
        self.cmb_backend.setCurrentText(new_backend_name)
        
        # 3. 準備新的
        backend_instance, kp_header = self._prepare_backend()
        if backend_instance is None:
            # 切換失敗，嘗試切換回來
            QtWidgets.QMessageBox.critical(self, "後端切換失敗", f"無法啟動 {new_backend_name} 後端，將嘗試切換回 {current_backend}")
            self.cmb_backend.setCurrentText(current_backend)
            backend_instance, kp_header = self._prepare_backend()
            
            if backend_instance is None:
                QtWidgets.QMessageBox.critical(self, "切換失敗", "兩個後端都無法啟動")
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                return

        # 4. 開啟新 CSV
        self.openCsv(kp_header)
        if self.csv_writer is None:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        # 5. 啟動新 Worker
        self.worker = CaptureWorker(
            cam_index=int(self.spn_cam.value()),
            backend=backend_instance
        )
        self.worker.frameReady.connect(self.onFrame)
        self.worker.rowReady.connect(self.onRow)
        self.accept_frames = True
        self.worker.start()

        # 6. 重啟計時
        self._start_elapsed_timer()

    @QtCore.pyqtSlot(QtGui.QImage, float)
    def onFrame(self, qimg, fps):
        if not self.accept_frames:
            return
        self.lbl_top_fps.setText(f"FPS: {fps:.1f}")
        self.canvas.setFps(fps)
        self.canvas.setFrame(qimg)

    @QtCore.pyqtSlot(list)
    def onRow(self, row):
        if self.csv_writer:
            self.csv_writer.writerow(row)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.onStop()
        e.accept()

# ==============================
def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    w = MainWindow()
    w.showMaximized()  # 或改成 showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
