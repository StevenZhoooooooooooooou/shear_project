#!/usr/bin/env python3
"""
Manual Control: Tactile + ArUco Vision Data Acquisition for Ball Catching
Features: Spacebar Quick Record, Backspace Quick Delete, ArUco Aperture Tracking
"""

import os
import json
import time
import threading
import queue
import numpy as np
import cv2
import serial
import tkinter as tk
from tkinter import messagebox
from collections import deque
from PIL import Image, ImageTk
from datetime import datetime
import atexit
import signal
import sys
import shutil  # 👉 新增：用于删除文件夹及其内容

# ============================================
# ⚙️ 环境变量与底层优化
# ============================================
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ==================== Configuration ====================
# [Camera & ArUco Config]
CAMERA_INDEX = 0

CAMERA_MATRIX = np.array([
    [929.95071746, 0.0, 560.65891142],
    [0.0, 931.15790397, 455.33871443],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array(
    [0.04275141, -0.07955943, 0.0007778, 0.00004574, 0.03054056],
    dtype=np.float64
)

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.021  # 2.1 cm 码的真实黑色边长
TARGET_ID_LEFT = 6
TARGET_ID_RIGHT = 7

OBJ_POINTS = np.array([
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# [Tactile]
TACTILE_PORT = "COM9"
TACTILE_BAUD = 12_000_000
TACTILE_N = 1024
TACTILE_THRES = 220
TACTILE_MAX_FORCE = 1023
TACTILE_SHAPE = (32, 32)
M_POINTS = 255
UI_SCALE = 9
UI_ARROW_SCALE = 3
TACTILE_Y_START = 0

# [Data Logging]
DATA_DIR = r"D:\Python_Project\shear_tactile_project\manual_control\data\ball_throwing"

# ==================== Global Async Logger & Variables ====================
tactile_running = True
latest_tactile_img = None
tactile_lock = threading.Lock()
stats = {'fps': 0, 'count': 0, 'times': deque(maxlen=100)}

shared_logger = {
    'recording': False,
    'start_time': 0,
    'data': {
        'tactile': [],
        'aperture': []
    }
}


# ============================================
# 🧠 空间欺骗：白边补全 ArUco 识别器
# ============================================
def detect_aruco_markers(frame, dict_type, pad_size=50):
    padded_frame = cv2.copyMakeBorder(
        frame, pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    gray = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)

    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(dict_type)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=cv2.aruco.DetectorParameters_create()
        )

    if corners is not None and len(corners) > 0:
        for i in range(len(corners)):
            corners[i][0][:, 0] -= pad_size
            corners[i][0][:, 1] -= pad_size

    return corners, ids


# ============================================
# 🚀 高速摄像头与 ArUco 融合线程
# ============================================
class FastCamera:
    def __init__(self, index):
        self.index = index
        self.cap, self.frame = None, None
        self.ret, self.running = False, False
        self.lock = threading.Lock()
        self.is_recording = False
        self.video_writer = None
        self.record_queue = queue.Queue(maxsize=300)
        self._open_camera()

    def _open_camera(self):
        if self.cap is not None:
            self.cap.release()
        print("🚀 正在通过 MSMF 协议连接摄像头...")
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
            self.cap.set(cv2.CAP_PROP_FPS, 100)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)
            return True
        return False

    def start(self):
        if not self.cap or not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        return True

    def _update(self):
        fail_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                fail_count = 0
                frame = cv2.flip(frame, -1)

                corners, ids = detect_aruco_markers(frame, ARUCO_DICT)
                aperture_cm = None

                cv2.rectangle(frame, (10, 10), (450, 130), (0, 0, 0), -1)

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    tvec_left, tvec_right = None, None
                    center_left, center_right = None, None

                    for i in range(len(ids)):
                        marker_id = ids[i][0]
                        marker_corners = corners[i][0]
                        if marker_id in [TARGET_ID_LEFT, TARGET_ID_RIGHT]:
                            success, _, tvec = cv2.solvePnP(
                                OBJ_POINTS, marker_corners, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE
                            )
                            if success:
                                cx, cy = int(np.mean(marker_corners[:, 0])), int(np.mean(marker_corners[:, 1]))
                                if marker_id == TARGET_ID_LEFT:
                                    tvec_left, center_left = tvec, (cx, cy)
                                    cv2.putText(frame, "L", (cx - 15, cy - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                                (255, 100, 100), 2)
                                elif marker_id == TARGET_ID_RIGHT:
                                    tvec_right, center_right = tvec, (cx, cy)
                                    cv2.putText(frame, "R", (cx - 15, cy - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                                (100, 255, 100), 2)

                    if tvec_left is not None and tvec_right is not None:
                        aperture_cm = np.linalg.norm(tvec_left - tvec_right) * 100
                        cv2.line(frame, center_left, center_right, (0, 255, 255), 3)
                        mid_x, mid_y = (center_left[0] + center_right[0]) // 2, (center_left[1] + center_right[1]) // 2
                        cv2.rectangle(frame, (mid_x - 60, mid_y - 30), (mid_x + 60, mid_y + 10), (0, 0, 0), -1)
                        cv2.putText(frame, f"{aperture_cm:.2f} cm", (mid_x - 55, mid_y), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                    (0, 255, 255), 2)

                        cv2.putText(frame, "STATUS: DUAL LOCK", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"APERTURE : {aperture_cm:5.2f} cm", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                    (0, 255, 255), 2)
                        z_diff = abs(tvec_left[2][0] - tvec_right[2][0]) * 100
                        cv2.putText(frame, f"Z-Skew   : {z_diff:5.2f} cm", (20, 115), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                    (200, 200, 200), 1)
                    else:
                        cv2.putText(frame, "Waiting for markers...", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                    (0, 165, 255), 2)
                else:
                    cv2.putText(frame, "Waiting for markers...", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255),
                                2)

                with self.lock:
                    self.ret, self.frame = True, frame

                if self.is_recording:
                    t = time.time() - shared_logger['start_time']
                    shared_logger['data']['aperture'].append({
                        't': round(t, 4),
                        'dist_cm': round(float(aperture_cm), 3) if aperture_cm is not None else None
                    })
                    if not self.record_queue.full():
                        self.record_queue.put(frame)
            else:
                fail_count += 1
                if fail_count > 50:
                    with self.lock:
                        self._open_camera()
                    fail_count = 0
                else:
                    time.sleep(0.005)

    def _writer_loop(self):
        while self.running:
            try:
                frame = self.record_queue.get(timeout=0.1)
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
            except queue.Empty:
                continue

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def start_recording(self, filepath):
        with self.lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
            else:
                w, h = 1280, 800
        self.video_writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 100.0, (w, h))
        self.record_queue.queue.clear()
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        time.sleep(0.2)
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        self.running = False
        self.stop_recording()
        if self.cap: self.cap.release()


# ==================== Tactile Sensor Processing ====================
def process_tactile(line_str):
    try:
        parts = line_str.split(',')
        n = len(parts) // 2
        if n < 1: return np.zeros(TACTILE_SHAPE, dtype=np.float32)
        positions = np.array(parts[:n], dtype=np.int32) - 1
        values = np.array(parts[n:2 * n], dtype=np.float32)
        vals = np.zeros(TACTILE_N, dtype=np.float32)
        valid = (positions >= 0) & (positions < TACTILE_N)
        vals[positions[valid]] = values[valid]
        img = vals.reshape(TACTILE_SHAPE, order="F")
        mask_geq = img >= TACTILE_THRES
        keep_mask = mask_geq.copy()
        keep_mask[1:, :] |= mask_geq[:-1, :]
        keep_mask[:-1, :] |= mask_geq[1:, :]
        keep_mask[:, 1:] |= mask_geq[:, :-1]
        keep_mask[:, :-1] |= mask_geq[:, 1:]
        img[~keep_mask] = 0
        return img
    except Exception:
        return None


def split_left_right(img_raw):
    imgL_raw = img_raw[:, 16:32]
    imgR_raw = img_raw[:, 0:16]
    imgLf = np.vstack((imgL_raw[:, 0:8], np.flipud(imgL_raw[:, 8:16])))
    imgRf = np.vstack((imgR_raw[:, 0:8], np.flipud(imgR_raw[:, 8:16])))
    return imgLf, imgRf


def split_left_right_quiver(flat_fx, flat_fy):
    fx_full = np.array(flat_fx, dtype=np.float32).reshape((16, 16), order='F')
    fy_full = np.array(flat_fy, dtype=np.float32).reshape((16, 16), order='F')
    fx_L_raw = fx_full[:, 8:16]
    fy_L_raw = fy_full[:, 8:16]
    fx_R_raw = fx_full[:, 0:8]
    fy_R_raw = fy_full[:, 0:8]
    fx_Lf = np.vstack((fx_L_raw[:, 0:4], np.flipud(fx_L_raw[:, 4:8])))
    fy_Lf = np.vstack((fy_L_raw[:, 0:4], -np.flipud(fy_L_raw[:, 4:8])))
    fx_Rf = np.vstack((fx_R_raw[:, 0:4], np.flipud(fx_R_raw[:, 4:8])))
    fy_Rf = np.vstack((fy_R_raw[:, 0:4], -np.flipud(fy_R_raw[:, 4:8])))
    return fx_Lf, fy_Lf, fx_Rf, fy_Rf


def calculate_shear_forces(img_matrix):
    H, W = img_matrix.shape
    blocks = img_matrix.reshape(H // 2, 2, W // 2, 2)
    left = blocks[:, :, :, 0].sum(axis=1)
    right = blocks[:, :, :, 1].sum(axis=1)
    top = blocks[:, 0, :, :].sum(axis=2)
    bottom = blocks[:, 1, :, :].sum(axis=2)
    total = blocks.sum(axis=(1, 3))
    denom = 4 * 1023.0 - total
    denom[np.abs(denom) < 1e-6] = 1e-6
    Fx = (right - left) / denom
    Fy = (bottom - top) / denom
    return Fx, Fy, total


def get_quiver_display_from_precomputed(patch_img, fx_mat, fy_mat, scale=9, arrow_scale=3):
    H, W = patch_img.shape
    norm = np.clip(patch_img / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    color = (color * 0.75).astype(np.uint8)
    disp = cv2.resize(color, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    blocks = patch_img.reshape(H // 2, 2, W // 2, 2)
    total = blocks.sum(axis=(1, 3))
    for i in range(H // 2):
        for j in range(W // 2):
            if total[i, j] < TACTILE_THRES * 1.5: continue
            fx, fy = fx_mat[i, j], fy_mat[i, j]
            if np.sqrt(fx ** 2 + fy ** 2) < 0.001: continue
            cx = int((0.5 + 2 * j) * scale)
            cy = int((0.5 + 2 * i) * scale)
            dx = int(fx * arrow_scale * scale)
            dy = int(fy * arrow_scale * scale)
            if abs(dx) > 1 or abs(dy) > 1:
                arrow_len = np.sqrt(dx ** 2 + dy ** 2)
                tip_len = min(8.0 / arrow_len, 0.4) if arrow_len > 1 else 0.3
                cv2.arrowedLine(disp, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2, tipLength=tip_len)
    return disp


def tactile_serial_reader():
    global tactile_running, latest_tactile_img, stats
    try:
        ser = serial.Serial(TACTILE_PORT, TACTILE_BAUD, timeout=0.001)
        ser.reset_input_buffer()
        time.sleep(0.5)
        ser.write(f"{M_POINTS}\n".encode('ascii'))
    except Exception as e:
        print(f"[TACTILE ERROR] {e}")
        tactile_running = False
        return

    buffer = b''
    while tactile_running:
        try:
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
            while b'\n' in buffer:
                idx = buffer.find(b'\n')
                line = buffer[:idx]
                buffer = buffer[idx + 1:]
                if len(line) > 5:
                    line_str = line.decode('ascii', errors='ignore').strip()
                    img = process_tactile(line_str)
                    if img is not None:
                        with tactile_lock:
                            latest_tactile_img = img
                            stats['count'] += 1
                            stats['times'].append(time.perf_counter())

                        if shared_logger['recording']:
                            t = time.time() - shared_logger['start_time']
                            flat_data = [round(float(v), 1) for v in img.flatten(order='F')]
                            fx_mat, fy_mat, _ = calculate_shear_forces(img)
                            flat_fx = [round(float(v), 4) for v in fx_mat.flatten(order='F')]
                            flat_fy = [round(float(v), 4) for v in fy_mat.flatten(order='F')]

                            shared_logger['data']['tactile'].append({
                                't': round(t, 4),
                                'v': flat_data,
                                'fx': flat_fx,
                                'fy': flat_fy
                            })
            if len(buffer) > 500000:
                buffer = buffer[-100000:]
            time.sleep(0.0001)
        except Exception:
            buffer = b''
            time.sleep(0.01)
    ser.close()


# ==================== Unified GUI Application ====================
class UnifiedRobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Control: Tactile + Vision + Aperture Tracking")
        self.root.geometry("1400x750")
        self.root.protocol("WM_DELETE_WINDOW", self.safe_shutdown)

        self.camera = None
        self.last_saved_dir = None  # 👉 新增：用于跟踪上一个保存的文件夹路径

        # 绑定快捷键：空格键录制，退格键删除
        self.root.bind("<space>", self.toggle_recording)
        self.root.bind("<BackSpace>", self.delete_last_session)  # 👉 新增：退格键删除绑定

        atexit.register(self.force_hardware_cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.init_hardware()
        self.build_ui()
        self.update_ui_loops()

    def signal_handler(self, sig, frame):
        print(f"\n[Signal {sig}] Caught forced exit. Cleaning up hardware...")
        self.force_hardware_cleanup()
        sys.exit(0)

    def force_hardware_cleanup(self):
        global tactile_running
        tactile_running = False
        try:
            self.stop_recording()
        except:
            pass
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass

    def safe_shutdown(self):
        self.force_hardware_cleanup()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        print("System safely shutdown.")

    def init_hardware(self):
        threading.Thread(target=tactile_serial_reader, daemon=True).start()
        self.camera = FastCamera(CAMERA_INDEX)
        if self.camera.start():
            print("✅ 摄像头与 ArUco 雷达已启动")
        else:
            print("⚠️ 摄像头连接失败")

    def build_ui(self):
        left_frame = tk.Frame(self.root, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tactile_frame = tk.Frame(left_frame, bg="black")
        tactile_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        tk.Label(tactile_frame, text="Tactile Sensor Feed", fg="white", bg="black", font=("Arial", 11, "bold")).pack(
            pady=2)
        self.tactile_label = tk.Label(tactile_frame, bg="black")
        self.tactile_label.pack(pady=5)

        camera_frame = tk.Frame(left_frame, bg="black")
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(camera_frame, text="Arducam + Aperture Radar", fg="white", bg="black",
                 font=("Arial", 11, "bold")).pack(pady=2)
        self.camera_label = tk.Label(camera_frame, bg="black")
        self.camera_label.pack(expand=True, fill=tk.BOTH, pady=5)

        right_frame = tk.Frame(self.root, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        rec_frame = tk.LabelFrame(right_frame, text="Data Logging Control", font=("Arial", 14, "bold"))
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(rec_frame, text="Press [SPACE] to Start/Stop\nPress [BACKSPACE] to Delete Last", fg="blue",
                 font=("Arial", 11, "italic")).pack(pady=15)

        self.btn_start = tk.Button(rec_frame, text="▶ START RECORD\n(Space)", font=("Arial", 16, "bold"), bg="#d4edda",
                                   command=self.start_recording, height=2)
        self.btn_start.pack(fill=tk.X, padx=20, pady=10)

        self.btn_stop = tk.Button(rec_frame, text="⏹ STOP RECORD\n(Space)", font=("Arial", 16, "bold"), bg="#f8d7da",
                                  command=self.stop_recording, state=tk.DISABLED, height=2)
        self.btn_stop.pack(fill=tk.X, padx=20, pady=10)

        # 👉 新增：专属的删除按钮
        self.btn_delete = tk.Button(rec_frame, text="🗑️ DELETE LAST\n(Backspace)", font=("Arial", 12, "bold"),
                                    bg="#ffcc00",
                                    command=self.delete_last_session, height=2)
        self.btn_delete.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_status = tk.Label(rec_frame, text="Standby", fg="gray", font=("Arial", 12))
        self.lbl_status.pack(pady=10)

    def update_ui_loops(self):
        with tactile_lock:
            img_raw = latest_tactile_img.copy() if latest_tactile_img is not None else None

        if img_raw is not None:
            left, right = split_left_right(img_raw)
            l_c = cv2.applyColorMap(np.clip(left / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
            r_c = cv2.applyColorMap(np.clip(right / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
            target_w, target_h = 8 * UI_SCALE, 64 * UI_SCALE
            l_s = cv2.resize(l_c, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            r_s = cv2.resize(r_c, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            fx_mat, fy_mat, _ = calculate_shear_forces(img_raw)
            flat_fx = [float(v) for v in fx_mat.flatten(order='F')]
            flat_fy = [float(v) for v in fy_mat.flatten(order='F')]
            fx_Lf, fy_Lf, fx_Rf, fy_Rf = split_left_right_quiver(flat_fx, flat_fy)

            left_force = get_quiver_display_from_precomputed(left, fx_Lf, fy_Lf, UI_SCALE, UI_ARROW_SCALE)
            right_force = get_quiver_display_from_precomputed(right, fx_Rf, fy_Rf, UI_SCALE, UI_ARROW_SCALE)

            sep = np.ones((target_h, 10, 3), dtype=np.uint8) * 50
            group_sep = np.ones((target_h, 30, 3), dtype=np.uint8) * 100
            combined = np.hstack([l_s, sep, r_s, group_sep, left_force, sep, right_force])

            self.tactile_label.imgtk = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)))
            self.tactile_label.configure(image=self.tactile_label.imgtk)

        if self.camera:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                display_frame = cv2.resize(frame, (768, 600))
                self.camera_label.imgtk_cam = ImageTk.PhotoImage(
                    image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
                self.camera_label.configure(image=self.camera_label.imgtk_cam)

        self.root.after(30, self.update_ui_loops)

    # ==== 录制功能 ====
    def toggle_recording(self, event=None):
        if shared_logger['recording']:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if shared_logger['recording']: return

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(DATA_DIR, f"session_{timestamp_str}")
        os.makedirs(self.session_dir, exist_ok=True)

        shared_logger['data'] = {'tactile': [], 'aperture': []}
        shared_logger['start_time'] = time.time()
        shared_logger['recording'] = True

        self.btn_start.config(state=tk.DISABLED, bg="#e9ecef")
        self.btn_stop.config(state=tk.NORMAL, bg="#dc3545", fg="white")
        self.lbl_status.config(text=f"🔴 RECORDING... ({timestamp_str})", fg="red", font=("Arial", 12, "bold"))

        if self.camera:
            video_path = os.path.join(self.session_dir, "camera_view.avi")
            self.camera.start_recording(video_path)

    def stop_recording(self):
        if not shared_logger['recording']: return

        shared_logger['recording'] = False
        self.btn_stop.config(state=tk.DISABLED, bg="#e9ecef", fg="black")
        self.lbl_status.config(text="Saving Files... Please wait", fg="#cc8800")
        self.root.update()

        if self.camera: self.camera.stop_recording()
        time.sleep(0.2)

        elapsed_time = time.time() - shared_logger['start_time']
        tac_len = len(shared_logger['data']['tactile'])
        apt_len = len(shared_logger['data']['aperture'])

        try:
            with open(os.path.join(self.session_dir, "sensor_data.json"), 'w', encoding='utf-8') as f:
                json.dump(shared_logger['data'], f)

            if elapsed_time > 0:
                meta_info = {
                    "session_info": {"duration_seconds": round(elapsed_time, 3)},
                    "hardware_success_reads": {"tactile_frames": tac_len, "aperture_frames": apt_len},
                    "hardware_frequencies_Hz": {
                        "tactile_sensor_hz": round(tac_len / elapsed_time, 2),
                        "aperture_tracker_hz": round(apt_len / elapsed_time, 2)
                    }
                }
                with open(os.path.join(self.session_dir, "meta_info.json"), 'w', encoding='utf-8') as mf:
                    json.dump(meta_info, mf, indent=4)

            # 👉 新增：保存成功后，记录当前路径供删除功能使用
            self.last_saved_dir = self.session_dir
            self.lbl_status.config(text=f"✅ Saved! Folder: {os.path.basename(self.session_dir)}", fg="green",
                                   font=("Arial", 11, "normal"))
            print(f"⏹ Session perfectly saved to: {self.session_dir}")
        except Exception as e:
            messagebox.showerror("Write Error", f"Cannot save JSON: {e}")
        finally:
            self.btn_start.config(state=tk.NORMAL, bg="#d4edda")

    # 👉 新增：一键删除上一组数据的核心逻辑
    def delete_last_session(self, event=None):
        if shared_logger['recording']:
            print("⚠️ 警告：正在录制中，无法执行删除操作！")
            return

        if self.last_saved_dir and os.path.exists(self.last_saved_dir):
            try:
                # 强力删除文件夹及其内部所有文件
                shutil.rmtree(self.last_saved_dir)
                folder_name = os.path.basename(self.last_saved_dir)

                # 界面提示删除成功
                self.lbl_status.config(text=f"🗑️ Deleted: {folder_name}", fg="red", font=("Arial", 11, "bold"))
                print(f"🗑️ 废弃数据已彻底删除: {self.last_saved_dir}")

                # 清空记录，防止连续按多次报错
                self.last_saved_dir = None
            except Exception as e:
                print(f"❌ 删除失败: {e}")
                messagebox.showerror("Delete Error", f"Failed to delete folder: {e}")
        else:
            print("⚠️ 没有找到刚保存的记录，或者已经被删除了。")
            self.lbl_status.config(text="⚠️ Nothing to delete right now.", fg="#cc8800")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = UnifiedRobotGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app.force_hardware_cleanup()
        except:
            pass