# 导入所需的库
import cv2
import json
import time
import tkinter as tk
from tkinter import messagebox, font as tkFont
from PIL import Image, ImageTk
import threading
import logging
import numpy as np
import datetime
import os
import platform
import mediapipe as mp
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg') # 在主线程外使用matplotlib需要这个
import matplotlib.pyplot as plt
import collections
import psutil # 用于获取系统信息
import torch # 确保torch已导入，用于设备检测

# --- 常量定义 ---
PERCLOS_WINDOW_DURATION = 60.0
MIN_FRAMES_FOR_PERCLOS = 10
FRAME_PROCESSING_INTERVAL = 3 # 每隔3帧跳过一次处理
YAWN_ALERT_THRESHOLD = 2 # 每两次打哈欠弹出警报
YAWN_COUNT_WINDOW_SECONDS = 60.0
FREQUENT_YAWN_ALARM_COOLDOWN_SECONDS = 60.0
PERCLOS_ALARM_COOLDOWN_SECONDS = 30.0

# --- Matplotlib 中文字体设置 ---
try:
    # 尝试使用常见的中文黑体和微软雅黑，以及Unicode Arial作为备选
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
except Exception as font_e:
    print(f"[WARN] 设置 Matplotlib 中文字体失败: {font_e}")
# --- 字体设置结束 ---


# --- 主应用程序类 ---
class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("疲劳驾驶检测")
        self.root.geometry("1300x850") # 设置窗口大小

        # --- 阈值定义 ---
        self.mouth_mar_threshold = 0.45
        self.dynamic_ear_closed_threshold = None
        self.current_perclos_trigger_threshold = 0.50 # 初始 PERCLOS 疲劳触发阈值

        # --- 日志和输出目录配置 ---
        log_suffix = f"PCLOS_CD{int(PERCLOS_ALARM_COOLDOWN_SECONDS)}_Yawn_CD{int(FREQUENT_YAWN_ALARM_COOLDOWN_SECONDS)}_MAR{int(self.mouth_mar_threshold*100)}"
        self.log_filename = f'疲劳检测日志_{log_suffix}.log'
        logging.basicConfig(filename=self.log_filename, level=logging.INFO,
                            format='%(asctime)s-%(levelname)s-%(threadName)s-%(filename)s:%(lineno)d-%(message)s',
                            encoding='utf-8', filemode='w')
        logging.info(f"日志将记录到: {self.log_filename}")
        logging.info(f"基于动态EAR的PERCLOS设置: 窗口={PERCLOS_WINDOW_DURATION}s, 初始阈值={self.current_perclos_trigger_threshold*100}%, 最小帧={MIN_FRAMES_FOR_PERCLOS}")
        logging.info(f"PERCLOS阈值将动态调整: 从校准完成开始, 初始50%, 300秒内线性降至20%")
        logging.info(f"PERCLOS警报冷却时间: {PERCLOS_ALARM_COOLDOWN_SECONDS} 秒")
        logging.info(f"动态闭眼阈值将基于: 校准睁眼EAR - 0.8 * (校准睁眼EAR - 校准闭眼EAR)")
        logging.info(f"帧处理间隔: 每 {FRAME_PROCESSING_INTERVAL} 帧处理一次")
        logging.info(f"MAR 打哈欠阈值: {self.mouth_mar_threshold}")
        logging.info(f"哈欠计数警报: {YAWN_ALERT_THRESHOLD} 次 / {YAWN_COUNT_WINDOW_SECONDS} 秒")
        logging.info(f"频繁哈欠警报冷却时间: {FREQUENT_YAWN_ALARM_COOLDOWN_SECONDS} 秒")
        print(f"日志文件路径: {os.path.abspath(self.log_filename)}")

        # --- 数据存储 ---
        self.timestamps_relative = []
        self.ear_values = []; self.mar_values = []; self.perclos_values = []
        self.perclos_threshold_plot_values = [] # 用于绘制 PERCLOS 动态阈值
        self.plot_lock = threading.Lock()
        self.yolo_processing_times = []
        self.mediapipe_processing_times = []
        self.fatigue_check_processing_times = []
        self.cpu_usage_values = []
        self.memory_usage_values = []

        # --- 疲劳状态 ---
        self.mouth_open_start_time = None; self.start_time = time.time()
        self.fatigue_events = []; self.last_warning_time = 0; self.warning_count = 0
        self.last_frequent_yawn_warning_time = 0
        self.last_perclos_alarm_time = 0
        self.processed_frame_timestamps = collections.deque(); self.closed_eye_timestamps = collections.deque()
        self.yawn_timestamps = collections.deque()
        self.yawn_count_in_minute = 0
        self.yawn_event_counted_for_current_opening = False

        # --- 校准变量 ---
        self.calibration_state = "INIT_OPEN_EYES"
        self.calibration_phase_start_time = None
        self.OPEN_EYES_CALIBRATION_DURATION = 10.0
        self.WAIT_FOR_CLOSE_DURATION = 5.0
        self.CLOSED_EYES_CALIBRATION_DURATION = 5.0
        self.ear_open_calibration_data = []; self.ear_closed_calibration_data = []
        self.ear_open_baseline = None; self.ear_closed_baseline = None
        self.calibration_completion_time = None
        self.FATIGUE_ALERT_GRACE_PERIOD_SECONDS = 20.0


        # --- GUI ---
        self.video_frame = tk.Frame(root); self.video_frame.pack(pady=10)
        self.canvas = tk.Canvas(self.video_frame, width=1280, height=720, bg='lightgrey'); self.canvas.pack()
        self.info_frame = tk.Frame(root); self.info_frame.pack(pady=10, fill=tk.X)
        default_font = tkFont.nametofont("TkDefaultFont")
        gui_label_font_size = default_font.actual()['size'] * 1.5
        # 修改：状态标签文字颜色为白色
        self.status_label = tk.Label(self.info_frame, text="检测状态: 准备校准...", font=(default_font.actual()['family'], int(gui_label_font_size)), width=60, anchor='w', fg="white"); self.status_label.pack(side=tk.LEFT, padx=10)
        self.warning_count_label = tk.Label(self.info_frame, text="警告次数: 0", font=(default_font.actual()['family'], int(gui_label_font_size*0.9)), width=15, anchor='w'); self.warning_count_label.pack(side=tk.LEFT, padx=5)
        self.perclos_label = tk.Label(self.info_frame, text="PERCLOS: - %", font=(default_font.actual()['family'], int(gui_label_font_size*0.9)), width=20, anchor='w'); self.perclos_label.pack(side=tk.LEFT, padx=5)

        # --- 初始化摄像头、模型等 ---
        self.cap = None; self.yolo_model = None; self.mp_face_mesh = None
        self.camera_fps = 0
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened(): raise ValueError("无法打开摄像头或摄像头不可用")
            self.camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.camera_fps == 0:
                logging.warning("摄像头返回FPS为0，可能无法获取准确帧率或摄像头FPS本身为0。")
            logging.info(f"摄像头初始化成功。摄像头报告的帧率 (FPS): {self.camera_fps:.2f}")
            print(f"摄像头报告的帧率 (FPS): {self.camera_fps:.2f}")
        except Exception as e: messagebox.showerror("摄像头错误", f"摄像头初始化失败: {e}"); logging.critical(f"摄像头初始化失败: {e}"); self.root.destroy(); return

        yolo_device = "cpu"
        try:
            self.yolo_model_name='yolov11n-face.pt' # 请确保此模型文件与脚本在同一目录或提供正确路径
            if not os.path.exists(self.yolo_model_name): raise FileNotFoundError(f"YOLO模型文件未找到: {self.yolo_model_name}")

            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                yolo_device = "mps"
                logging.info("MPS is available and built, YOLO将尝试在MPS上运行。")
                print("MPS is available and built, YOLO将尝试在MPS上运行。")
            elif torch.cuda.is_available():
                yolo_device = "cuda"
                logging.info("CUDA is available, YOLO将尝试在CUDA上运行。")
                print("CUDA is available, YOLO将尝试在CUDA上运行。")
            else:
                logging.info("MPS 和 CUDA 都不可用, YOLO将在CPU上运行。")
                print("MPS 和 CUDA 都不可用, YOLO将在CPU上运行。")
            
            self.yolo_model = YOLO(self.yolo_model_name)
            self.yolo_model.to(yolo_device)
            logging.info(f"成功加载YOLO模型: '{self.yolo_model_name}' 并配置在 '{yolo_device}' 设备上运行。")
            print(f"成功加载YOLO模型: '{self.yolo_model_name}' 并配置在 '{yolo_device}' 设备上运行。")

        except Exception as e:
            messagebox.showerror("YOLO模型错误", f"加载或配置YOLO模型到设备 '{yolo_device}' 失败: {e}")
            logging.critical(f"加载或配置YOLO模型到设备 '{yolo_device}' 失败: {e}")
            print(f"加载或配置YOLO模型到设备 '{yolo_device}' 失败，将回退到CPU。")
            yolo_device = "cpu"
            try:
                if self.yolo_model is None:
                    self.yolo_model = YOLO(self.yolo_model_name)
                self.yolo_model.to(yolo_device)
                logging.info(f"成功加载YOLO模型: '{self.yolo_model_name}' 并在CPU上运行。")
                print(f"成功加载YOLO模型: '{self.yolo_model_name}' 并在CPU上运行。")
            except Exception as e_cpu:
                 messagebox.showerror("YOLO模型错误", f"在CPU上加载YOLO模型也失败: {e_cpu}")
                 logging.critical(f"在CPU上加载YOLO模型也失败: {e_cpu}")
                 self.root.destroy()
                 return

        try:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.202)
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
            logging.info("MediaPipe FaceMesh 初始化成功 (refine_landmarks=True)")
        except Exception as e: messagebox.showerror("MediaPipe错误", f"MediaPipe初始化失败: {e}"); logging.critical(f"MediaPipe初始化失败: {e}"); self.root.destroy(); return

        self.stop_event = threading.Event(); self.detection_thread = threading.Thread(target=self.run_detection, name="DetectionThread", daemon=True); self.current_frame = None; self.detection_lock = threading.Lock(); self.detection_thread.start(); logging.info("后台检测线程启动")
        self.update_ui(); self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def play_alert_sound(self, frequency=1000, duration=300):
        system_platform = platform.system()
        try:
            if system_platform == "Windows": import winsound; winsound.Beep(frequency, duration); logging.debug(f"播放提示音(winsound)")
            elif system_platform == "Darwin":
                return_code = os.system(f'afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1')
                if return_code != 0: logging.warning(f"macOS afplay播放失败 (返回码:{return_code})，尝试BEL字符。"); print('\a', end='', flush=True)
                else: logging.debug("播放提示音(afplay)")
            else:
                print('\a', end='', flush=True); logging.debug("播放提示音(BEL字符)")
        except Exception as e: logging.warning(f"播放提示音失败: {e}"); print('\a', end='', flush=True)

    def show_calibration_results_and_sound(self):
        open_baseline_str = f"{self.ear_open_baseline:.4f}" if self.ear_open_baseline is not None else "失败"
        closed_baseline_str = f"{self.ear_closed_baseline:.4f}" if self.ear_closed_baseline is not None else "失败"
        dynamic_thr_str = f"{self.dynamic_ear_closed_threshold:.4f}" if self.dynamic_ear_closed_threshold is not None else "未设定"
        message = (f"校准完成！\n\n"
                   f"睁眼EAR基准(平均值): {open_baseline_str}\n"
                   f"闭眼EAR基准(平均值): {closed_baseline_str}\n"
                   f"动态闭眼阈值(PERCLOS用): {dynamic_thr_str}\n\n"
                   f"开始疲劳检测。")
        logging.info(f"显示校准结果: 睁眼基准={open_baseline_str}, 闭眼基准={closed_baseline_str}, 动态闭眼阈值={dynamic_thr_str}")
        self.root.after(0, lambda: messagebox.showinfo("校准完成", message, parent=self.root))
        self.play_alert_sound(frequency=1200, duration=400)

    def parse_and_check_fatigue(self, right_eye_ear_value, mouth_ratio):
        if self.calibration_state != "COMPLETE": return
        current_time = time.time(); fatigue_detected_this_cycle = False; calculated_perclos = 0.0
        mouth_duration_threshold = 2.0 # 本次哈欠持续时间阈值为2s

        if self.calibration_completion_time is not None:
            time_since_calibration = current_time - self.calibration_completion_time
            if time_since_calibration <= 0: 
                self.current_perclos_trigger_threshold = 0.50
            elif 0 < time_since_calibration <= 300: 
                progress = time_since_calibration / 300.0
                self.current_perclos_trigger_threshold = 0.50 - progress * (0.50 - 0.20) 
                self.current_perclos_trigger_threshold = max(0.20, self.current_perclos_trigger_threshold)
            else: 
                self.current_perclos_trigger_threshold = 0.20
        else: 
            self.current_perclos_trigger_threshold = 0.50

        window_start_time = current_time - PERCLOS_WINDOW_DURATION
        while self.processed_frame_timestamps and self.processed_frame_timestamps[0] < window_start_time: self.processed_frame_timestamps.popleft()
        while self.closed_eye_timestamps and self.closed_eye_timestamps[0] < window_start_time: self.closed_eye_timestamps.popleft()
        num_processed = len(self.processed_frame_timestamps); num_closed = len(self.closed_eye_timestamps)

        if num_processed >= MIN_FRAMES_FOR_PERCLOS:
            calculated_perclos = num_closed / num_processed if num_processed > 0 else 0.0
            dyn_thresh_str = f"{self.dynamic_ear_closed_threshold:.3f}" if self.dynamic_ear_closed_threshold is not None else 'N/A'
            logging.info(f"动态EAR-PERCLOS: 闭眼帧={num_closed}, 总处理帧={num_processed}, PERCLOS={calculated_perclos:.3f} (当前EAR闭眼阈值:{dyn_thresh_str}, 当前疲劳触发PERCLOS阈值:{self.current_perclos_trigger_threshold:.2f})")
            if calculated_perclos >= self.current_perclos_trigger_threshold:
                if (current_time - self.last_perclos_alarm_time) >= PERCLOS_ALARM_COOLDOWN_SECONDS:
                    warning_reason = f"PERCLOS过高({calculated_perclos*100:.1f}%, 阈值 {self.current_perclos_trigger_threshold*100:.1f}%)"
                    if self.trigger_fatigue_warning(warning_reason):
                        self.last_perclos_alarm_time = current_time
                        fatigue_detected_this_cycle = True
                else:
                    logging.debug(f"PERCLOS过高条件满足, 但处于 {PERCLOS_ALARM_COOLDOWN_SECONDS}秒 冷却中.")
        else: calculated_perclos = -1
        
        display_perclos = f"{calculated_perclos*100:.1f}%" if calculated_perclos >= 0 else "- %"
        self.root.after(0, lambda p=display_perclos, t=self.current_perclos_trigger_threshold: self.perclos_label.config(text=f"PERCLOS: {p} (Th:{t*100:.0f}%)"))

        a_yawn_was_confirmed_this_cycle = False
        if not np.isnan(mouth_ratio) and mouth_ratio > self.mouth_mar_threshold:
            if self.mouth_open_start_time is None:
                self.mouth_open_start_time = current_time
            elif not self.yawn_event_counted_for_current_opening and \
                 (current_time - self.mouth_open_start_time >= mouth_duration_threshold):
                self.yawn_timestamps.append(current_time)
                self.yawn_event_counted_for_current_opening = True
                a_yawn_was_confirmed_this_cycle = True
                logging.info(f"检测到并记录一次哈欠事件 @ {datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
        else:
            self.mouth_open_start_time = None
            self.yawn_event_counted_for_current_opening = False

        yawn_window_start_time = current_time - YAWN_COUNT_WINDOW_SECONDS
        while self.yawn_timestamps and self.yawn_timestamps[0] < yawn_window_start_time:
            self.yawn_timestamps.popleft()
        self.yawn_count_in_minute = len(self.yawn_timestamps)

        if self.yawn_count_in_minute >= YAWN_ALERT_THRESHOLD: # YAWN_ALERT_THRESHOLD is already 2
            if self.calibration_state == "COMPLETE" and not fatigue_detected_this_cycle:
                if (current_time - self.last_frequent_yawn_warning_time) >= FREQUENT_YAWN_ALARM_COOLDOWN_SECONDS:
                    warning_reason = f"频繁哈欠 ({self.yawn_count_in_minute}次/{int(YAWN_COUNT_WINDOW_SECONDS)}秒)"
                    if self.trigger_fatigue_warning(warning_reason):
                        self.last_frequent_yawn_warning_time = current_time
                        fatigue_detected_this_cycle = True
                else:
                    logging.debug(f"频繁哈欠条件 ({self.yawn_count_in_minute}次)满足, 但处于 {FREQUENT_YAWN_ALARM_COOLDOWN_SECONDS}秒 冷却中.")

        if not fatigue_detected_this_cycle:
            if time.time() - self.last_warning_time > 2.0:
                # 修改：状态标签文字颜色为白色
                self.root.after(0, lambda: self.status_label.config(text="检测状态: 正常", fg="white"))

        with self.plot_lock:
            if self.timestamps_relative:
                current_plot_perclos_thresh = self.current_perclos_trigger_threshold if self.calibration_state == "COMPLETE" else np.nan
                while len(self.perclos_values) < len(self.timestamps_relative) -1 : self.perclos_values.append(np.nan)
                self.perclos_values.append(calculated_perclos if calculated_perclos >= 0 else np.nan)
                while len(self.perclos_threshold_plot_values) < len(self.timestamps_relative) -1 : self.perclos_threshold_plot_values.append(np.nan)
                self.perclos_threshold_plot_values.append(current_plot_perclos_thresh if calculated_perclos >=0 else np.nan)

    def trigger_fatigue_warning(self, reason):
        current_time = time.time()
        if self.calibration_completion_time is not None:
            time_since_calibration = current_time - self.calibration_completion_time
            if time_since_calibration < self.FATIGUE_ALERT_GRACE_PERIOD_SECONDS:
                logging.info(f"抑制警告 '{reason}' (原因: 校准后 {self.FATIGUE_ALERT_GRACE_PERIOD_SECONDS}秒 宽限期内)"); return False
        general_warning_cooldown = 5.0
        if current_time - self.last_warning_time < general_warning_cooldown:
            logging.debug(f"抑制警告 '{reason}' (原因: 全局冷却中 - {general_warning_cooldown}s). 上次警告时间: {self.last_warning_time:.2f}, 当前时间: {current_time:.2f}")
            return False
        self.play_alert_sound(frequency=1500, duration=500)
        self.last_warning_time = current_time
        self.warning_count += 1
        self.root.after(0, lambda c=self.warning_count: self.warning_count_label.config(text=f"警告次数: {c}"))
        # 修改：状态标签文字颜色为白色 (疲劳时)
        self.root.after(0, lambda r=reason: self.status_label.config(text=f"检测状态: 疲劳 ({r})", fg="white"))
        event_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        event_data = {"time": event_time_str, "reason": reason, "duration_since_start_seconds": round(current_time - self.start_time, 2), "warning_count_total": self.warning_count}
        if "PERCLOS" in reason:
            event_data["perclos_value_at_trigger"] = self.perclos_values[-1] if self.perclos_values and not np.isnan(self.perclos_values[-1]) else "N/A"
            event_data["perclos_threshold_at_trigger"] = self.current_perclos_trigger_threshold
        self.fatigue_events.append(event_data); logging.warning(f"疲劳警报触发: {reason} @ {event_time_str}")
        def show_warning_popup():
            try:
                warning_window = tk.Toplevel(self.root); warning_window.title("疲劳警报!"); warning_window.geometry("450x200"); warning_window.attributes('-topmost', True); warning_window.transient(self.root)
                popup_font_size = tkFont.nametofont("TkDefaultFont").actual()['size'] * 1.8
                tk.Label(warning_window, text=f"检测到疲劳!\n原因: {reason}\n请立即注意休息!", wraplength=420, justify=tk.CENTER, fg="red", font=('Arial', int(popup_font_size), 'bold')).pack(pady=30, padx=10)
                warning_window.after(5000, warning_window.destroy)
            except Exception as e: logging.error(f"显示警告弹窗时出错: {e}")
        self.root.after(0, show_warning_popup)
        return True

    def run_detection(self):
        frame_count = 0; processed_frame_count = 0
        yolo_conf_threshold = 0.5; face_class_id = 0
        RIGHT_EYE_EAR_TOP_INDEX = 159; RIGHT_EYE_EAR_BOTTOM_INDEX = 145
        RIGHT_EYE_EAR_LEFT_INDEX = 133; RIGHT_EYE_EAR_RIGHT_INDEX = 33
        MOUTH_TOP_INDEX = 13; MOUTH_BOTTOM_INDEX = 14
        MOUTH_LEFT_INDEX = 78; MOUTH_RIGHT_INDEX = 308
        required_ids_for_detection = set([
            RIGHT_EYE_EAR_TOP_INDEX, RIGHT_EYE_EAR_BOTTOM_INDEX, RIGHT_EYE_EAR_LEFT_INDEX, RIGHT_EYE_EAR_RIGHT_INDEX,
            MOUTH_TOP_INDEX, MOUTH_BOTTOM_INDEX, MOUTH_LEFT_INDEX, MOUTH_RIGHT_INDEX
        ])
        yolo_time_ms = np.nan; mediapipe_time_ms = np.nan; fatigue_check_time_ms = np.nan
        cpu_percent = np.nan; memory_mb = np.nan
        while not self.stop_event.is_set():
            ret, frame = self.cap.read(); frame_count += 1
            if not ret or frame is None: logging.error(f"无法从摄像头读取帧 (帧号 {frame_count})"); time.sleep(0.1); continue
            if frame_count % FRAME_PROCESSING_INTERVAL != 0:
                if frame is not None:
                     with self.detection_lock: self.current_frame = frame.copy()
                time.sleep(0.01); continue
            current_processing_timestamp = time.time()
            processed_frame_count += 1
            frame_height, frame_width, _ = frame.shape; frame_display = frame.copy()
            expanded_face_roi = None; expanded_roi_coords = None
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.percent
            except Exception as e: logging.warning(f"获取系统资源信息失败: {e}"); cpu_percent = np.nan; memory_mb = np.nan
            
            yolo_start_time = time.time()
            try:
                yolo_results = self.yolo_model(frame, verbose=False, conf=yolo_conf_threshold, classes=[face_class_id])
                if yolo_results and yolo_results[0].boxes:
                    boxes_data = yolo_results[0].boxes.cpu().numpy()
                    if len(boxes_data.xyxy) > 0:
                        best_box_index = np.argmax(boxes_data.conf)
                        box = boxes_data[best_box_index]
                        x1,y1,x2,y2 = map(int,box.xyxy[0])
                        center_x, center_y = (x1+x2)/2, (y1+y2)/2
                        roi_width, roi_height = (x2-x1), (y2-y1)
                        scale_factor = 1.2 # 将ROI扩大的比例设置为20%
                        exp_w, exp_h = roi_width * scale_factor, roi_height * scale_factor
                        exp_x1 = max(0, int(center_x - exp_w / 2)); exp_y1 = max(0, int(center_y - exp_h / 2))
                        exp_x2 = min(frame_width, int(center_x + exp_w / 2)); exp_y2 = min(frame_height, int(center_y + exp_h / 2))
                        if exp_x1 < exp_x2 and exp_y1 < exp_y2 and (exp_x2-exp_x1) > 20 and (exp_y2-exp_y1) > 20:
                            expanded_face_roi = frame[exp_y1:exp_y2, exp_x1:exp_x2]
                            expanded_roi_coords = (exp_x1, exp_y1, exp_x2-exp_x1, exp_y2-exp_y1)
                            cv2.rectangle(frame_display,(x1,y1),(x2,y2),(255,0,0),1)
                            cv2.rectangle(frame_display,(exp_x1,exp_y1),(exp_x2,exp_y2),(0,255,255),2)
            except Exception as e: logging.error(f"YOLO人脸检测或ROI提取过程中出错: {e}", exc_info=True)
            yolo_end_time = time.time(); yolo_time_ms = (yolo_end_time - yolo_start_time) * 1000
            if self.stop_event.is_set(): break
            right_eye_ear_value = np.nan; mouth_ratio = np.nan; ear_based_closed_this_frame = False
            mediapipe_start_time = time.time()
            if expanded_face_roi is not None and expanded_roi_coords is not None:
                results_mp = None
                try:
                    face_roi_rgb = cv2.cvtColor(expanded_face_roi, cv2.COLOR_BGR2RGB)
                    face_roi_rgb.flags.writeable = False; results_mp = self.mp_face_mesh.process(face_roi_rgb); face_roi_rgb.flags.writeable = True
                except Exception as e: logging.error(f"MediaPipe处理人脸ROI时出错: {e}", exc_info=True)
                if self.stop_event.is_set(): break
                if results_mp and results_mp.multi_face_landmarks:
                    face_landmarks = results_mp.multi_face_landmarks[0]; landmark_coords_global = {}
                    num_total_landmarks = len(face_landmarks.landmark); roi_x_offset, roi_y_offset, roi_w, roi_h = expanded_roi_coords
                    for landmark_id in required_ids_for_detection:
                        if 0 <= landmark_id < num_total_landmarks:
                            lm = face_landmarks.landmark[landmark_id]
                            global_x = int(lm.x * roi_w + roi_x_offset); global_y = int(lm.y * roi_h + roi_y_offset)
                            landmark_coords_global[landmark_id] = (global_x, global_y)
                        else: landmark_coords_global[landmark_id] = None
                    p_top = landmark_coords_global.get(RIGHT_EYE_EAR_TOP_INDEX); p_bottom = landmark_coords_global.get(RIGHT_EYE_EAR_BOTTOM_INDEX)
                    p_left = landmark_coords_global.get(RIGHT_EYE_EAR_LEFT_INDEX); p_right = landmark_coords_global.get(RIGHT_EYE_EAR_RIGHT_INDEX)
                    if p_top and p_bottom and p_left and p_right:
                        ver_dist = np.linalg.norm(np.array(p_top) - np.array(p_bottom)); hor_dist = np.linalg.norm(np.array(p_left) - np.array(p_right))
                        right_eye_ear_value = ver_dist / hor_dist if hor_dist > 1e-6 else 0.0
                    m_top = landmark_coords_global.get(MOUTH_TOP_INDEX); m_bottom = landmark_coords_global.get(MOUTH_BOTTOM_INDEX)
                    m_left = landmark_coords_global.get(MOUTH_LEFT_INDEX); m_right = landmark_coords_global.get(MOUTH_RIGHT_INDEX)
                    if m_top and m_bottom and m_left and m_right:
                        ver_dist_mouth = np.linalg.norm(np.array(m_top) - np.array(m_bottom)); hor_dist_mouth = np.linalg.norm(np.array(m_left) - np.array(m_right))
                        mouth_ratio = ver_dist_mouth / hor_dist_mouth if hor_dist_mouth > 1e-6 else 0.0
                    if self.calibration_state=="COMPLETE" and self.dynamic_ear_closed_threshold is not None:
                        if not np.isnan(right_eye_ear_value) and right_eye_ear_value < self.dynamic_ear_closed_threshold and right_eye_ear_value > 1e-9:
                            ear_based_closed_this_frame = True
                    if self.calibration_state=="COMPLETE" and not np.isnan(right_eye_ear_value):
                        self.processed_frame_timestamps.append(current_processing_timestamp)
                        if ear_based_closed_this_frame: self.closed_eye_timestamps.append(current_processing_timestamp)
                else:
                    if self.calibration_state == "COMPLETE":
                        self.mouth_open_start_time = None; self.yawn_event_counted_for_current_opening = False
                        if time.time() - self.last_warning_time > 2.0:
                             # 修改：状态标签文字颜色为白色
                            self.root.after(0, lambda: self.status_label.config(text="检测状态: 未检测到面部关键点", fg="white"))
                            self.root.after(0, lambda: self.perclos_label.config(text="PERCLOS: - %"))
            elif not (expanded_face_roi and expanded_roi_coords):
                 if self.calibration_state == "COMPLETE":
                     self.mouth_open_start_time = None; self.yawn_event_counted_for_current_opening = False
                     if time.time() - self.last_warning_time > 2.0:
                         # 修改：状态标签文字颜色为白色
                         self.root.after(0, lambda: self.status_label.config(text="检测状态: 未检测到人脸", fg="white"))
                         self.root.after(0, lambda: self.perclos_label.config(text="PERCLOS: - %"))
            mediapipe_end_time = time.time(); mediapipe_time_ms = (mediapipe_end_time - mediapipe_start_time) * 1000

            if self.calibration_state == "INIT_OPEN_EYES":
                self.calibration_phase_start_time = current_processing_timestamp
                self.calibration_state = "CALIBRATING_OPEN_EYES"; self.ear_open_calibration_data = []
                logging.info(f"睁眼校准开始 ({self.OPEN_EYES_CALIBRATION_DURATION:.0f}秒)..."); self.play_alert_sound(frequency=600, duration=150)
                # 修改：状态标签文字颜色为白色
                self.root.after(0, lambda: self.status_label.config(text=f"检测状态: 睁眼校准 ({self.OPEN_EYES_CALIBRATION_DURATION:.0f}秒)... 请自然睁眼", fg="white"))
            elif self.calibration_state == "CALIBRATING_OPEN_EYES":
                elapsed_time = current_processing_timestamp - self.calibration_phase_start_time
                if elapsed_time <= self.OPEN_EYES_CALIBRATION_DURATION:
                    if not np.isnan(right_eye_ear_value) and right_eye_ear_value > 1e-9: self.ear_open_calibration_data.append(right_eye_ear_value)
                    remaining_time = self.OPEN_EYES_CALIBRATION_DURATION - elapsed_time
                    # 修改：状态标签文字颜色为白色
                    self.root.after(0, lambda rt=remaining_time: self.status_label.config(text=f"检测状态: 睁眼校准 ({max(0,rt):.0f}秒剩余)...", fg="white") if self.calibration_state == "CALIBRATING_OPEN_EYES" else None)
                else:
                    if self.ear_open_calibration_data:
                        valid_ears = [ear for ear in self.ear_open_calibration_data if ear > 0.1]
                        if valid_ears: self.ear_open_baseline = np.mean(valid_ears); logging.info(f"睁眼校准完成。EAR基准(平均值): {self.ear_open_baseline:.4f}")
                        else: self.ear_open_baseline = None; logging.warning("睁眼校准：收集到的有效EAR数据不足。")
                    else: self.ear_open_baseline = None; logging.warning("睁眼校准：未收集到EAR数据。")
                    self.play_alert_sound(frequency=800, duration=200)
                    self.calibration_state = "WAIT_FOR_CLOSE"; self.calibration_phase_start_time = current_processing_timestamp
                    # 修改：状态标签文字颜色为白色
                    self.root.after(0, lambda: self.status_label.config(text=f"检测状态: 请准备闭眼 ({self.WAIT_FOR_CLOSE_DURATION:.0f}秒)...", fg="white"))
            elif self.calibration_state == "WAIT_FOR_CLOSE":
                elapsed_time = current_processing_timestamp - self.calibration_phase_start_time
                if elapsed_time <= self.WAIT_FOR_CLOSE_DURATION:
                    remaining_time = self.WAIT_FOR_CLOSE_DURATION - elapsed_time
                    # 修改：状态标签文字颜色为白色
                    self.root.after(0, lambda rt=remaining_time: self.status_label.config(text=f"检测状态: ({max(0,rt):.0f}秒后) 请闭眼...", fg="white") if self.calibration_state == "WAIT_FOR_CLOSE" else None)
                else: self.calibration_state = "INIT_CLOSED_EYES"; self.play_alert_sound(frequency=700, duration=150)
            elif self.calibration_state == "INIT_CLOSED_EYES":
                self.calibration_phase_start_time = current_processing_timestamp
                self.calibration_state = "CALIBRATING_CLOSED_EYES"; self.ear_closed_calibration_data = []
                logging.info(f"闭眼校准开始 ({self.CLOSED_EYES_CALIBRATION_DURATION:.0f}秒)...");
                # 修改：状态标签文字颜色为白色
                self.root.after(0, lambda: self.status_label.config(text=f"检测状态: 闭眼校准 ({self.CLOSED_EYES_CALIBRATION_DURATION:.0f}秒)... 请闭眼", fg="white"))
            elif self.calibration_state == "CALIBRATING_CLOSED_EYES":
                elapsed_time = current_processing_timestamp - self.calibration_phase_start_time
                if elapsed_time <= self.CLOSED_EYES_CALIBRATION_DURATION:
                    if not np.isnan(right_eye_ear_value): self.ear_closed_calibration_data.append(right_eye_ear_value)
                    remaining_time = self.CLOSED_EYES_CALIBRATION_DURATION - elapsed_time
                    # 修改：状态标签文字颜色为白色
                    self.root.after(0, lambda rt=remaining_time: self.status_label.config(text=f"检测状态: 闭眼校准 ({max(0,rt):.0f}秒剩余)...", fg="white") if self.calibration_state == "CALIBRATING_CLOSED_EYES" else None)
                else:
                    if self.ear_closed_calibration_data:
                        valid_closed_ears = [ear for ear in self.ear_closed_calibration_data if ear < 0.2 and ear > 1e-9]
                        if valid_closed_ears: self.ear_closed_baseline = np.mean(valid_closed_ears); logging.info(f"闭眼校准完成。EAR基准(平均值): {self.ear_closed_baseline:.4f}")
                        else: self.ear_closed_baseline = None; logging.warning("闭眼校准（EAR）：收集到的有效数据不足。")
                    else: self.ear_closed_baseline = None; logging.warning("闭眼校准（EAR）：未收集到数据。")
                    if self.ear_open_baseline is not None and self.ear_closed_baseline is not None and self.ear_open_baseline > self.ear_closed_baseline + 1e-3:
                        threshold = self.ear_open_baseline - 0.8 * (self.ear_open_baseline - self.ear_closed_baseline)
                        threshold = max(self.ear_closed_baseline + 0.01, threshold)
                        threshold = min(self.ear_open_baseline - 0.01, threshold)
                        self.dynamic_ear_closed_threshold = threshold
                        logging.info(f"动态闭眼阈值(PERCLOS用)成功设定为: {self.dynamic_ear_closed_threshold:.4f} (基于公式: Open - 0.8*(Open-Closed))")
                    else:
                        self.dynamic_ear_closed_threshold = None
                        logging.warning("无法计算动态闭眼阈值，原因：睁眼/闭眼基准校准失败，或睁眼EAR不大于闭眼EAR。将无法使用基于动态阈值的PERCLOS判断。")
                    self.calibration_state = "COMPLETE"; logging.info("所有校准阶段完成。")
                    self.calibration_completion_time = current_processing_timestamp
                    # 修改：状态标签文字颜色为白色
                    self.root.after(0, lambda: self.status_label.config(text="检测状态: 校准完成! 开始疲劳检测...", fg="white"))
                    self.show_calibration_results_and_sound()
                    # 修改：状态标签文字颜色为白色
                    self.root.after(4000, lambda: self.status_label.config(text="检测状态: 正常", fg="white") if self.calibration_state == "COMPLETE" and time.time() - self.last_warning_time > 2.0 else None)

            fatigue_check_start_time = time.time()
            if self.calibration_state == "COMPLETE":
                self.parse_and_check_fatigue(right_eye_ear_value, mouth_ratio)
            fatigue_check_end_time = time.time()
            fatigue_check_time_ms = (fatigue_check_end_time - fatigue_check_start_time) * 1000 if self.calibration_state == "COMPLETE" else 0

            current_time_relative = current_processing_timestamp - self.start_time
            with self.plot_lock:
                self.timestamps_relative.append(current_time_relative)
                self.ear_values.append(right_eye_ear_value)
                self.mar_values.append(mouth_ratio)
                self.yolo_processing_times.append(yolo_time_ms if yolo_time_ms > 0 else np.nan)
                self.mediapipe_processing_times.append(mediapipe_time_ms if mediapipe_time_ms > 0 and expanded_face_roi is not None else np.nan)
                self.fatigue_check_processing_times.append(fatigue_check_time_ms if fatigue_check_time_ms > 0 and self.calibration_state == "COMPLETE" else np.nan)
                self.cpu_usage_values.append(cpu_percent if not np.isnan(cpu_percent) else np.nan)
                self.memory_usage_values.append(memory_mb if not np.isnan(memory_mb) else np.nan)

            if self.calibration_state == "COMPLETE":
                text_yawn_count = f"Yawns (last {int(YAWN_COUNT_WINDOW_SECONDS)}s): {self.yawn_count_in_minute}"
                screen_font_scale = 1.0; screen_font_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text_yawn_count, cv2.FONT_HERSHEY_SIMPLEX, screen_font_scale, screen_font_thickness)
                y_pos_text = frame_display.shape[0] - 20; x_pos_text = frame_display.shape[1] - text_width - 20
                cv2.putText(frame_display, text_yawn_count, (x_pos_text, y_pos_text), cv2.FONT_HERSHEY_SIMPLEX, screen_font_scale, (0, 0, 255), screen_font_thickness, cv2.LINE_AA)
            with self.detection_lock: self.current_frame = frame_display.copy()
        logging.info("检测线程已退出主循环。")

    def update_ui(self):
        frame_to_show = None
        with self.detection_lock:
            if self.current_frame is not None: frame_to_show = self.current_frame.copy()
        if frame_to_show is not None:
            try:
                img_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
                if canvas_width > 1 and canvas_height > 1:
                    img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    self.img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            except Exception as e: logging.error(f"UI 更新图像时出错: {e}", exc_info=True)
        if not self.stop_event.is_set(): self.root.after(33, self.update_ui)
        else: logging.info("UI 更新循环已停止。")

    def on_closing(self):
        logging.info("接收到关闭应用程序的请求...")
        if self.stop_event.is_set(): logging.warning("程序已在关闭过程中，忽略重复请求。"); return
        self.stop_event.set()
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            logging.info("等待检测线程结束...")
            self.detection_thread.join(timeout=5.0)
            if self.detection_thread.is_alive(): logging.warning("检测线程在超时后仍未结束。")
            else: logging.info("检测线程已成功结束。")
        logging.info("开始生成数据图表...")
        with self.plot_lock:
            time_rel_copy = list(self.timestamps_relative); ear_copy = list(self.ear_values); mar_copy = list(self.mar_values)
            perclos_copy = list(self.perclos_values); perclos_thresh_plot_copy = list(self.perclos_threshold_plot_values)
            yolo_times_copy = list(self.yolo_processing_times); mediapipe_times_copy = list(self.mediapipe_processing_times)
            fatigue_times_copy = list(self.fatigue_check_processing_times); cpu_usage_copy = list(self.cpu_usage_values)
            memory_usage_copy = list(self.memory_usage_values)
        min_len = len(time_rel_copy)
        if min_len == 0: logging.warning("无有效的时间戳数据，无法生成任何图表。")
        else:
            def align_data(data_list, target_len):
                current_len = len(data_list)
                if current_len > target_len: return data_list[:target_len]
                elif current_len < target_len: return data_list + [np.nan] * (target_len - current_len)
                return data_list
            ear_copy = align_data(ear_copy, min_len); mar_copy = align_data(mar_copy, min_len)
            perclos_copy = align_data(perclos_copy, min_len); perclos_thresh_plot_copy = align_data(perclos_thresh_plot_copy, min_len)
            yolo_times_copy = align_data(yolo_times_copy, min_len); mediapipe_times_copy = align_data(mediapipe_times_copy, min_len)
            fatigue_times_copy = align_data(fatigue_times_copy, min_len); cpu_usage_copy = align_data(cpu_usage_copy, min_len)
            memory_usage_copy = align_data(memory_usage_copy, min_len)
            title_fontsize = 24; label_fontsize = 20; legend_fontsize = 16; tick_fontsize = 14
            if any(not np.isnan(x) for x in ear_copy):
                try:
                    fig_ear, ax_ear = plt.subplots(figsize=(14, 7))
                    valid_indices_ear = [i for i, x in enumerate(ear_copy) if not np.isnan(x)]
                    if valid_indices_ear: ax_ear.plot(np.array(time_rel_copy)[valid_indices_ear], np.array(ear_copy)[valid_indices_ear], marker='.', linestyle='-', markersize=4, linewidth=1, label='EAR (右眼)')
                    if self.dynamic_ear_closed_threshold is not None: ax_ear.axhline(y=self.dynamic_ear_closed_threshold, color='orange', linestyle='--', linewidth=1.5, label=f'动态闭眼阈值(校准) ({self.dynamic_ear_closed_threshold:.3f})')
                    if self.ear_open_baseline is not None: ax_ear.axhline(y=self.ear_open_baseline, color='g', linestyle=':', linewidth=1.5, label=f'校准睁眼基准(均值) ({self.ear_open_baseline:.3f})')
                    if self.ear_closed_baseline is not None: ax_ear.axhline(y=self.ear_closed_baseline, color='magenta', linestyle=':', linewidth=1.5, label=f'校准闭眼基准(均值) ({self.ear_closed_baseline:.3f})')
                    ax_ear.set_xlabel("时间 (秒)", fontsize=label_fontsize); ax_ear.set_ylabel("EAR 值", fontsize=label_fontsize)
                    ax_ear.set_title("右眼EAR值随时间变化趋势", fontsize=title_fontsize); ax_ear.grid(True, linestyle='--', alpha=0.7)
                    ax_ear.legend(fontsize=legend_fontsize); ax_ear.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    plt.tight_layout(); plot_filename_ear = "EAR_plot.png"; plt.savefig(plot_filename_ear)
                    logging.info(f"EAR 图表已保存至: {os.path.abspath(plot_filename_ear)}")
                except Exception as plot_e: logging.error(f"生成 EAR 图表失败: {plot_e}", exc_info=True)
                finally:
                    if 'fig_ear' in locals() and fig_ear is not None : plt.close(fig_ear)
            else: logging.warning("无有效的 EAR 数据用于绘图。")
            if any(not np.isnan(x) for x in mar_copy):
                try:
                    fig_mar, ax_mar = plt.subplots(figsize=(14, 7))
                    valid_indices_mar = [i for i, x in enumerate(mar_copy) if not np.isnan(x)]
                    if valid_indices_mar: ax_mar.plot(np.array(time_rel_copy)[valid_indices_mar], np.array(mar_copy)[valid_indices_mar], marker='.', linestyle='-', markersize=4, linewidth=1, label='MAR (嘴部)')
                    ax_mar.axhline(y=self.mouth_mar_threshold, color='b', linestyle='--', linewidth=1.5, label=f'打哈欠MAR阈值 ({self.mouth_mar_threshold})')
                    ax_mar.set_xlabel("时间 (秒)", fontsize=label_fontsize); ax_mar.set_ylabel("MAR 值", fontsize=label_fontsize)
                    ax_mar.set_title("嘴部MAR值随时间变化趋势", fontsize=title_fontsize); ax_mar.grid(True, linestyle='--', alpha=0.7)
                    ax_mar.legend(fontsize=legend_fontsize); ax_mar.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    plt.tight_layout(); plot_filename_mar = "MAR_plot.png"; plt.savefig(plot_filename_mar)
                    logging.info(f"MAR 图表已保存至: {os.path.abspath(plot_filename_mar)}")
                except Exception as plot_e: logging.error(f"生成 MAR 图表失败: {plot_e}", exc_info=True)
                finally:
                    if 'fig_mar' in locals() and fig_mar is not None : plt.close(fig_mar)
            else: logging.warning("无有效的 MAR 数据用于绘图。")
            if any(not np.isnan(x) and x >=0 for x in perclos_copy):
                try:
                    fig_perclos, ax_perclos = plt.subplots(figsize=(14, 7))
                    valid_indices_perclos = [i for i, x in enumerate(perclos_copy) if not np.isnan(x) and x >= 0]
                    if valid_indices_perclos: ax_perclos.plot(np.array(time_rel_copy)[valid_indices_perclos], np.array(perclos_copy)[valid_indices_perclos] * 100, marker='.', linestyle='-', markersize=4, linewidth=1, label=f'动态EAR-PERCLOS ({PERCLOS_WINDOW_DURATION}s窗口)')
                    valid_indices_thresh = [i for i, x in enumerate(perclos_thresh_plot_copy) if not np.isnan(x)]
                    if valid_indices_thresh: ax_perclos.plot(np.array(time_rel_copy)[valid_indices_thresh], np.array(perclos_thresh_plot_copy)[valid_indices_thresh] * 100, color='r', linestyle='--', linewidth=1.5, label='动态疲劳PERCLOS阈值 (%)')
                    else: ax_perclos.axhline(y=0.5 * 100, color='r', linestyle=':', linewidth=1, label='初始疲劳阈值 (50%)'); ax_perclos.axhline(y=0.2 * 100, color='darkred', linestyle=':', linewidth=1, label='最终疲劳阈值 (20%)')
                    ax_perclos.set_xlabel("时间 (秒)", fontsize=label_fontsize); ax_perclos.set_ylabel("PERCLOS (%)", fontsize=label_fontsize)
                    ax_perclos.set_title("动态EAR-PERCLOS随时间变化趋势", fontsize=title_fontsize); ax_perclos.grid(True, linestyle='--', alpha=0.7)
                    ax_perclos.legend(fontsize=legend_fontsize); ax_perclos.set_ylim(0, 105); ax_perclos.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    plt.tight_layout(); plot_filename_perclos = "dynamic_ear_perclos_plot.png"; plt.savefig(plot_filename_perclos)
                    logging.info(f"动态EAR-PERCLOS 图表已保存至: {os.path.abspath(plot_filename_perclos)}")
                except Exception as plot_e: logging.error(f"生成 PERCLOS 图表失败: {plot_e}", exc_info=True)
                finally:
                    if 'fig_perclos' in locals() and fig_perclos is not None : plt.close(fig_perclos)
            else: logging.warning("无有效的 PERCLOS 数据用于绘图。")
            if any(not np.isnan(x) for x in yolo_times_copy) or any(not np.isnan(x) for x in mediapipe_times_copy) or any(not np.isnan(x) for x in fatigue_times_copy):
                try:
                    fig_proc_time, ax_proc_time = plt.subplots(figsize=(14, 7))
                    valid_yolo = [(t, val) for t, val in zip(time_rel_copy, yolo_times_copy) if not np.isnan(val)];
                    if valid_yolo: ax_proc_time.plot([item[0] for item in valid_yolo], [item[1] for item in valid_yolo], marker='.', linestyle='-', markersize=3, linewidth=0.8, label='YOLO检测耗时 (ms)')
                    valid_mp = [(t, val) for t, val in zip(time_rel_copy, mediapipe_times_copy) if not np.isnan(val)];
                    if valid_mp: ax_proc_time.plot([item[0] for item in valid_mp], [item[1] for item in valid_mp], marker='.', linestyle='-', markersize=3, linewidth=0.8, label='MediaPipe提取耗时 (ms)')
                    valid_fatigue = [(t, val) for t, val in zip(time_rel_copy, fatigue_times_copy) if not np.isnan(val) and val > 0];
                    if valid_fatigue: ax_proc_time.plot([item[0] for item in valid_fatigue], [item[1] for item in valid_fatigue], marker='.', linestyle='-', markersize=3, linewidth=0.8, label='疲劳判断逻辑耗时 (ms)')
                    ax_proc_time.set_xlabel("时间 (秒)", fontsize=label_fontsize); ax_proc_time.set_ylabel("处理时间 (毫秒)", fontsize=label_fontsize)
                    ax_proc_time.set_title("各阶段处理时间消耗", fontsize=title_fontsize); ax_proc_time.grid(True, linestyle='--', alpha=0.7)
                    ax_proc_time.legend(fontsize=legend_fontsize); ax_proc_time.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    plt.tight_layout(); plot_filename_proc_time = "processing_times_plot.png"; plt.savefig(plot_filename_proc_time)
                    logging.info(f"处理时间图表已保存至: {os.path.abspath(plot_filename_proc_time)}")
                except Exception as plot_e: logging.error(f"生成处理时间图表失败: {plot_e}", exc_info=True)
                finally:
                    if 'fig_proc_time' in locals() and fig_proc_time is not None: plt.close(fig_proc_time)
            else: logging.warning("无有效的处理时间数据用于绘图。")
            if any(not np.isnan(x) for x in cpu_usage_copy) or any(not np.isnan(x) for x in memory_usage_copy):
                try:
                    fig_sys, ax_cpu = plt.subplots(figsize=(14, 7)); color_cpu = 'tab:red'
                    ax_cpu.set_xlabel("时间 (秒)", fontsize=label_fontsize); ax_cpu.set_ylabel("CPU 使用率 (%)", color=color_cpu, fontsize=label_fontsize)
                    valid_cpu_indices = [i for i, x in enumerate(cpu_usage_copy) if not np.isnan(x)]
                    if valid_cpu_indices: ax_cpu.plot(np.array(time_rel_copy)[valid_cpu_indices], np.array(cpu_usage_copy)[valid_cpu_indices], color=color_cpu, marker='.', linestyle='-', markersize=3, linewidth=0.8, label='CPU 使用率 (%)')
                    ax_cpu.tick_params(axis='y', labelcolor=color_cpu, labelsize=tick_fontsize); ax_cpu.tick_params(axis='x', labelsize=tick_fontsize); ax_cpu.set_ylim(0, 105)
                    ax_mem = ax_cpu.twinx(); color_mem = 'tab:blue'
                    ax_mem.set_ylabel("内存使用率 (%)", color=color_mem, fontsize=label_fontsize)
                    valid_mem_indices = [i for i, x in enumerate(memory_usage_copy) if not np.isnan(x)]
                    if valid_mem_indices: ax_mem.plot(np.array(time_rel_copy)[valid_mem_indices], np.array(memory_usage_copy)[valid_mem_indices], color=color_mem, marker='.', linestyle='--', markersize=3, linewidth=0.8, label='内存使用率 (%)')
                    ax_mem.tick_params(axis='y', labelcolor=color_mem, labelsize=tick_fontsize); ax_mem.set_ylim(0, 105)
                    fig_sys.suptitle("系统资源使用情况 (CPU 和内存百分比)", fontsize=title_fontsize)
                    lines_cpu, labels_cpu = ax_cpu.get_legend_handles_labels(); lines_mem, labels_mem = ax_mem.get_legend_handles_labels()
                    ax_mem.legend(lines_cpu + lines_mem, labels_cpu + labels_mem, loc='upper right', fontsize=legend_fontsize)
                    ax_cpu.grid(True, linestyle=':', alpha=0.6)
                    plt.tight_layout(rect=[0, 0, 1, 0.96]); plot_filename_sys = "system_usage_plot.png"; plt.savefig(plot_filename_sys)
                    logging.info(f"系统资源使用图表已保存至: {os.path.abspath(plot_filename_sys)}")
                except Exception as plot_e: logging.error(f"生成系统资源使用图表失败: {plot_e}", exc_info=True)
                finally:
                    if 'fig_sys' in locals() and fig_sys is not None: plt.close(fig_sys)
            else: logging.warning("无有效的CPU或内存数据用于绘图。")
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened(): self.cap.release(); logging.info("摄像头已成功释放。")
        if self.fatigue_events:
            log_path_json = f"疲劳事件记录_{self.log_filename.split('_', 2)[-1].replace('.log', '.json')}"
            try:
                with open(log_path_json, "w", encoding='utf-8') as f: json.dump(self.fatigue_events, f, ensure_ascii=False, indent=4)
                logging.info(f"所有疲劳事件已保存至JSON文件: {os.path.abspath(log_path_json)}")
            except Exception as e: logging.error(f"保存疲劳事件JSON日志失败: {e}")
        else: logging.info("程序运行期间未记录疲劳事件。")
        try:
            if self.root: self.root.destroy(); logging.info("应用程序窗口已成功关闭。")
        except Exception as e: logging.error(f"关闭Tkinter主窗口时发生错误: {e}")

if __name__ == "__main__":
    initial_log_file = f"FATAL_ERROR_STARTUP_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    app_instance_log_file = initial_log_file
    try:
        root = tk.Tk()
        app = DetectionApp(root)
        app_instance_log_file = app.log_filename
        root.mainloop()
    except Exception as main_loop_exception:
        if not logging.getLogger().hasHandlers(): # type: ignore
            logging.basicConfig(filename=app_instance_log_file, level=logging.CRITICAL, format='%(asctime)s-%(levelname)s-%(message)s', encoding='utf-8', filemode='a')
        logging.critical(f"应用程序主循环外部或初始化期间发生严重错误: {main_loop_exception}", exc_info=True)
        try:
            error_display_root = tk.Tk(); error_display_root.withdraw()
            messagebox.showerror("严重错误", f"程序启动或运行时发生无法恢复的严重错误:\n{main_loop_exception}\n\n详情请查看日志文件:\n'{os.path.abspath(app_instance_log_file)}'", parent=None)
            error_display_root.destroy()
        except Exception as display_error:
            print(f"CRITICAL ERROR (UNABLE TO SHOW IN GUI): {main_loop_exception}")
            print(f"GUI display error: {display_error}")
            print(f"Please check log file: {os.path.abspath(app_instance_log_file)}")