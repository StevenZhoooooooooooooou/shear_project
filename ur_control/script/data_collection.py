#!/usr/bin/env python3
"""
Tactile + Camera Multi-Modal Data Visualizer & Logger
Features: Dynamic Video Resolution Fix, Side-by-Side Visuals, Session Folder Saving, No Software Limits
"""

import os

# 1. 强行屏蔽 OpenCV 底层的 C++ 警告输出
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
# 2. 强行禁用 MSMF 的硬件加速转换，解决连接卡顿
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import json
import time
import math
import threading
import queue
import numpy as np
import cv2
import serial
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque
from PIL import Image, ImageTk
from datetime import datetime
import atexit
import signal
import sys

# ==== 导入硬件库 ====
try:
    from dynamixel_api import XL430W250TConnector
    from package.motor_control import ActuatedUMI
except ImportError:
    print("Warning: Gripper libraries not found. Run in simulation mode.")

try:
    from package.ur_robot import URRobot, Axis
except ImportError:
    print("Warning: UR Robot library not found.")

# ==================== Configuration ====================
# [Camera]
CAMERA_INDEX = 0

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

# [Gripper]
GRIPPER_PORT = "COM8"
GRIPPER_BAUD = 57600
DYNAMIXEL_ID = 1

# [UR Robot]
ROBOT_IP = "192.168.1.102"
JOG_CONT_STEP_M = 0.01
ROT_CONT_STEP_DEG = 2.0
EE_ROTATION_DIR = 1

# [Data Logging]
DATA_DIR = r"D:\Python_Project\shear_tactile_project\robot_arm\data"

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
        'ur': [],
        'gripper': []
    }
}


# ==================== 高速摄像头线程类 ====================
class FastCamera:
    """整合视频录制的高速 MSMF 摄像头线程"""

    def __init__(self, index):
        self.index = index
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()

        # 录制相关
        self.is_recording = False
        self.video_writer = None
        self.record_queue = queue.Queue(maxsize=300)

        # 硬编码曝光为 -7
        self.current_exposure = -7
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
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exposure)
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

                with self.lock:
                    self.ret, self.frame = True, frame

                if self.is_recording:
                    if not self.record_queue.full():
                        self.record_queue.put(frame)
            else:
                fail_count += 1
                if fail_count > 50:
                    print("⚠️ 摄像头断开，尝试热重启...")
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
        # 【核心修复】：动态获取当前摄像头的真实物理分辨率，绝不硬编码
        with self.lock:
            if self.frame is not None:
                h, w = self.frame.shape[:2]
            else:
                # 后备分辨率
                w, h = 1280, 800

                # 使用真实获取到的 (w, h) 初始化录制器
        self.video_writer = cv2.VideoWriter(
            filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 100.0, (w, h)
        )
        self.record_queue.queue.clear()
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        time.sleep(0.5)
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
    if TACTILE_Y_START > 0:
        imgLf[:TACTILE_Y_START, :] = 0
        imgRf[:TACTILE_Y_START, :] = 0
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
        self.root.title("Multi-Modal Robot Data Acquisition System")
        self.root.geometry("1650x850")
        self.root.protocol("WM_DELETE_WINDOW", self.safe_shutdown)

        self.ur_robot = None
        self.connector = None
        self.gripper = None
        self.camera = None

        self.gripper_lock = threading.Lock()
        self.last_cmd_time = 0
        self.active_jog_axis = None
        self.active_jog_dir = 0
        self.active_rot_dir = 0
        self.speed_control_failed = False

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
        except Exception:
            pass

        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass

        if self.ur_robot:
            try:
                print("Releasing UR Robot RTDE registers...")
                if hasattr(self.ur_robot, 'rtde_c') and not self.speed_control_failed:
                    self.ur_robot.rtde_c.speedStop(10.0)
            except Exception:
                pass

            try:
                self.ur_robot.disconnect()
            except Exception:
                pass
            self.ur_robot = None

        if self.gripper or self.connector:
            try:
                print("Releasing Gripper Torque and Serial Port...")
                if self.gripper:
                    self.gripper.torque_enabled = False
            except Exception as e:
                print(f"Torque release fail (Ignored): {e}")

            try:
                if self.connector:
                    self.connector.disconnect()
            except Exception as e:
                print(f"Serial disconnect fail (Ignored): {e}")

            self.gripper = None
            self.connector = None

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
            print("✅ 摄像头已连接")
        else:
            print("⚠️ 摄像头连接失败")

        try:
            print(f"Connecting to UR Robot at {ROBOT_IP}...")
            self.ur_robot = URRobot({"robot_ip": ROBOT_IP, "velocity": 0.2, "acceleration": 0.2})
            print("✅ UR Robot Connected.")
        except Exception as e:
            print(f"⚠️ UR Robot Connection Failed: {e}")

        try:
            self.connector = XL430W250TConnector(device=GRIPPER_PORT, baud_rate=GRIPPER_BAUD, dynamixel_id=DYNAMIXEL_ID)
            self.connector.connect()
            self.gripper = ActuatedUMI(self.connector)
            self.gripper.torque_enabled = False
            self.gripper.operating_mode = 3
            self.connector.write_field("shutdown", 0)
            self.gripper.torque_enabled = True
            self.connector.write_field("profile_velocity", 50)
            print("✅ Gripper Connected.")
        except Exception as e:
            print(f"⚠️ Gripper Connection Failed: {e}")

    def build_ui(self):
        # 左右并排容器
        left_frame = tk.Frame(self.root, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 触觉子面板 (左) ---
        tactile_frame = tk.Frame(left_frame, bg="black")
        tactile_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        tk.Label(tactile_frame, text="Tactile Sensor Feed", fg="white", bg="black", font=("Arial", 11, "bold")).pack(
            pady=2)
        self.tactile_label = tk.Label(tactile_frame, bg="black")
        self.tactile_label.pack(pady=5)

        # --- 摄像头子面板 (右) ---
        camera_frame = tk.Frame(left_frame, bg="black")
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(camera_frame, text="Arducam Video Feed (Exp: -7, Flipped)", fg="white", bg="black",
                 font=("Arial", 11, "bold")).pack(pady=2)
        self.camera_label = tk.Label(camera_frame, bg="black")
        self.camera_label.pack(expand=True, fill=tk.BOTH, pady=5)

        # 右侧控制区
        right_frame = tk.Frame(self.root, width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ur_frame = tk.LabelFrame(right_frame, text="UR Robot Control", font=("Arial", 11, "bold"))
        ur_frame.pack(fill=tk.X, pady=10, ipady=5)

        jog_frame = tk.Frame(ur_frame)
        jog_frame.pack(side=tk.LEFT, padx=15)

        def create_jog_btn(parent, text, bg, axis, direction, row, col):
            btn = tk.Button(parent, text=text, bg=bg, width=6, height=1, font=("Arial", 10, "bold"))
            btn.grid(row=row, column=col, padx=3, pady=3)
            btn.bind("<ButtonPress-1>", lambda e, a=axis, d=direction: self.start_jog(a, d))
            btn.bind("<ButtonRelease-1>", self.stop_motion)
            btn.bind("<Leave>", self.stop_motion)
            return btn

        create_jog_btn(jog_frame, "X +", "#e2e3e5", Axis.X, 1, 0, 0)
        create_jog_btn(jog_frame, "X -", "#e2e3e5", Axis.X, -1, 0, 1)
        create_jog_btn(jog_frame, "Y +", "#d1ecf1", Axis.Y, 1, 1, 0)
        create_jog_btn(jog_frame, "Y -", "#d1ecf1", Axis.Y, -1, 1, 1)
        create_jog_btn(jog_frame, "Z +", "#d4edda", Axis.Z, 1, 2, 0)
        create_jog_btn(jog_frame, "Z -", "#d4edda", Axis.Z, -1, 2, 1)

        rot_frame = tk.Frame(ur_frame)
        rot_frame.pack(side=tk.RIGHT, padx=25)
        tk.Label(rot_frame, text="EE Rotation (Wrist 3)", font=("Arial", 10, "bold")).pack(pady=5)
        btn_rot_frame = tk.Frame(rot_frame)
        btn_rot_frame.pack(pady=5)
        btn_ccw = tk.Button(btn_rot_frame, text="↺ CCW", bg="#ffcccc", font=("Arial", 10, "bold"), width=8)
        btn_ccw.pack(side=tk.LEFT, padx=5)
        btn_ccw.bind("<ButtonPress-1>", lambda e: self.start_rotate(1))
        btn_ccw.bind("<ButtonRelease-1>", self.stop_motion)
        btn_ccw.bind("<Leave>", self.stop_motion)
        btn_cw = tk.Button(btn_rot_frame, text="↻ CW", bg="#ccccff", font=("Arial", 10, "bold"), width=8)
        btn_cw.pack(side=tk.LEFT, padx=5)
        btn_cw.bind("<ButtonPress-1>", lambda e: self.start_rotate(-1))
        btn_cw.bind("<ButtonRelease-1>", self.stop_motion)
        btn_cw.bind("<Leave>", self.stop_motion)

        grp_frame = tk.LabelFrame(right_frame, text="Gripper Control", font=("Arial", 11, "bold"))
        grp_frame.pack(fill=tk.X, pady=10)
        tk.Label(grp_frame, text="Speed (0-265):").pack(pady=2)
        self.speed_slider = tk.Scale(grp_frame, from_=10, to_=265, orient=tk.HORIZONTAL, length=300,
                                     command=self.update_gripper_speed)
        self.speed_slider.set(50)
        self.speed_slider.pack()
        tk.Label(grp_frame, text="Position (0-4095):").pack(pady=2)
        self.pos_slider = tk.Scale(grp_frame, from_=0, to_=4095, orient=tk.HORIZONTAL, length=300,
                                   command=self.update_gripper_pos)
        try:
            self.pos_slider.set(self.connector.read_field("present_position") if self.connector else 2048)
        except:
            self.pos_slider.set(2048)
        self.pos_slider.pack()

        rec_frame = tk.LabelFrame(right_frame, text="Multi-Sensor Session Logging", font=("Arial", 11, "bold"))
        rec_frame.pack(fill=tk.X, pady=10)
        self.btn_start = tk.Button(rec_frame, text="▶ Start Session", font=("Arial", 10, "bold"), bg="#d4edda",
                                   command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=10, pady=10)
        self.btn_stop = tk.Button(rec_frame, text="⏹ Stop Session", font=("Arial", 10, "bold"), bg="#f8d7da",
                                  command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10, pady=10)
        self.lbl_status = tk.Label(rec_frame, text="Standby", fg="gray")
        self.lbl_status.pack(pady=10)

    def stop_motion(self, event=None):
        self.active_jog_axis = None
        self.active_jog_dir = 0
        self.active_rot_dir = 0

    def start_jog(self, axis, direction):
        if not self.ur_robot: return
        self.active_jog_axis = axis
        self.active_jog_dir = direction
        threading.Thread(target=self._jog_loop, daemon=True).start()

    def _jog_loop(self):
        axis = self.active_jog_axis
        direction = self.active_jog_dir
        speed = 0.05
        vel_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if axis == Axis.X:
            vel_vector[0] = direction * speed
        elif axis == Axis.Y:
            vel_vector[1] = direction * speed
        elif axis == Axis.Z:
            vel_vector[2] = direction * speed

        while self.active_jog_axis == axis and self.active_jog_dir == direction:
            try:
                if hasattr(self.ur_robot, 'rtde_c') and not getattr(self, 'speed_control_failed', False):
                    try:
                        self.ur_robot.rtde_c.speedL(vel_vector, 1.0, 0.002)
                    except Exception:
                        self.speed_control_failed = True
                if getattr(self, 'speed_control_failed', True) or not hasattr(self.ur_robot, 'rtde_c'):
                    self.ur_robot.move_in_axis_rel(axis, direction * JOG_CONT_STEP_M)
                    time.sleep(0.05)
            except Exception:
                break

        if hasattr(self.ur_robot, 'rtde_c') and not getattr(self, 'speed_control_failed', False):
            try:
                self.ur_robot.rtde_c.speedStop(10.0)
            except Exception:
                pass

    def start_rotate(self, direction):
        if not self.ur_robot: return
        self.active_rot_dir = direction
        threading.Thread(target=self._rotate_loop, daemon=True).start()

    def _rotate_loop(self):
        direction = self.active_rot_dir
        rot_speed = 0.5

        while self.active_rot_dir == direction:
            try:
                joint_vel = [0.0, 0.0, 0.0, 0.0, 0.0, direction * rot_speed * EE_ROTATION_DIR]
                if hasattr(self.ur_robot, 'rtde_c') and not getattr(self, 'speed_control_failed', False):
                    try:
                        self.ur_robot.rtde_c.speedJ(joint_vel, 2.0, 0.002)
                    except Exception:
                        self.speed_control_failed = True

                if getattr(self, 'speed_control_failed', True) or not hasattr(self.ur_robot, 'rtde_c'):
                    joints = list(self.ur_robot.current_joint_positions)
                    joints[5] += np.deg2rad(direction * ROT_CONT_STEP_DEG * EE_ROTATION_DIR)
                    self.ur_robot.move_to_joint_position(joints)
                    time.sleep(0.05)
            except Exception:
                break

        if hasattr(self.ur_robot, 'rtde_c') and not getattr(self, 'speed_control_failed', False):
            try:
                self.ur_robot.rtde_c.speedStop(10.0)
            except Exception:
                pass

    def update_gripper_speed(self, val):
        if not self.connector: return
        t = time.time()
        if t - self.last_cmd_time > 0.05:
            try:
                with self.gripper_lock:
                    self.connector.write_field_async("profile_velocity", int(val))
                self.last_cmd_time = t
            except:
                pass

    def update_gripper_pos(self, val):
        if not self.connector: return
        t = time.time()
        if t - self.last_cmd_time > 0.05:
            try:
                with self.gripper_lock:
                    self.connector.write_field_async("goal_position", int(val))
                self.last_cmd_time = t
            except:
                pass

    # ==== UI 更新循环 ====
    def update_ui_loops(self):
        with tactile_lock:
            img_raw = latest_tactile_img.copy() if latest_tactile_img is not None else None

        if img_raw is not None:
            left, right = split_left_right(img_raw)
            l_c = cv2.applyColorMap(np.clip(left / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
            r_c = cv2.applyColorMap(np.clip(right / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
            target_w = 8 * UI_SCALE
            target_h = 64 * UI_SCALE
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

            rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.tactile_label.imgtk = imgtk
            self.tactile_label.configure(image=imgtk)

        if self.camera:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                display_frame = cv2.resize(frame, (768, 600))
                rgb_cam = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_cam = Image.fromarray(rgb_cam)
                imgtk_cam = ImageTk.PhotoImage(image=pil_cam)
                self.camera_label.imgtk = imgtk_cam
                self.camera_label.configure(image=imgtk_cam)

        self.root.after(30, self.update_ui_loops)

    # ==== 录制功能 ====
    def start_recording(self):
        if shared_logger['recording']: return

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(DATA_DIR, f"session_{timestamp_str}")
        os.makedirs(self.session_dir, exist_ok=True)

        shared_logger['data'] = {'tactile': [], 'ur': [], 'gripper': []}
        shared_logger['start_time'] = time.time()
        shared_logger['recording'] = True

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"🔴 Recording Session to: session_{timestamp_str}", fg="red")

        if self.camera:
            video_path = os.path.join(self.session_dir, "camera_view.avi")
            self.camera.start_recording(video_path)

        threading.Thread(target=self.ur_record_loop, daemon=True).start()
        threading.Thread(target=self.gripper_record_loop, daemon=True).start()

    def stop_recording(self):
        if not shared_logger['recording']: return

        shared_logger['recording'] = False
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Saving Session Files... Please wait", fg="#cc8800")
        self.root.update()

        if self.camera:
            self.camera.stop_recording()

        time.sleep(0.2)
        elapsed_time = time.time() - shared_logger['start_time']
        tac_len = len(shared_logger['data']['tactile'])
        ur_len = len(shared_logger['data']['ur'])
        grp_len = len(shared_logger['data']['gripper'])

        data_filepath = os.path.join(self.session_dir, "sensor_data.json")
        meta_filepath = os.path.join(self.session_dir, "meta_info.json")

        try:
            with open(data_filepath, 'w', encoding='utf-8') as f:
                json.dump(shared_logger['data'], f)

            if elapsed_time > 0:
                meta_info = {
                    "session_info": {"duration_seconds": round(elapsed_time, 3)},
                    "hardware_success_reads": {"tactile_frames": tac_len, "ur_robot_reads": ur_len,
                                               "gripper_reads": grp_len},
                    "hardware_frequencies_Hz": {
                        "tactile_sensor_hz": round(tac_len / elapsed_time, 2),
                        "ur_robot_hz": round(ur_len / elapsed_time, 2),
                        "gripper_hz": round(grp_len / elapsed_time, 2),
                        "camera_fps": "100.0 (Target)"
                    }
                }
                with open(meta_filepath, 'w', encoding='utf-8') as mf:
                    json.dump(meta_info, mf, indent=4)

            self.lbl_status.config(text=f"✅ Session Saved! Folder: {os.path.basename(self.session_dir)}", fg="green")
            print(f"⏹ Session files perfectly saved to: {self.session_dir}")

        except Exception as e:
            messagebox.showerror("Write Error", f"Cannot save JSON: {e}")
        finally:
            self.btn_start.config(state=tk.NORMAL)

    def ur_record_loop(self):
        while shared_logger['recording']:
            try:
                if self.ur_robot:
                    t = time.time() - shared_logger['start_time']
                    pos = self.ur_robot.end_effector_position
                    ori = self.ur_robot.end_effector_orientation
                    shared_logger['data']['ur'].append({
                        't': round(t, 4), 'p': [round(x, 5) for x in pos], 'o': [round(x, 5) for x in ori]
                    })
            except Exception:
                pass
            time.sleep(0.01)

    def gripper_record_loop(self):
        while shared_logger['recording']:
            try:
                if self.connector:
                    t = time.time() - shared_logger['start_time']
                    with self.gripper_lock:
                        time.sleep(0.001)
                        pos = self.connector.read_field("present_position")
                    shared_logger['data']['gripper'].append({'t': round(t, 4), 'pos': pos})
            except Exception:
                pass
            time.sleep(0.033)


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