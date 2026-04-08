#!/usr/bin/env python3
"""
Tactile Sensor Data Visualizer with Progress Bar
Reads and visualizes tactile data from JSON files, including pressure and shear force directions.
"""

import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

# ==================== Configuration ====================
TACTILE_SHAPE = (32, 32)
TACTILE_MAX_FORCE = 1023
TACTILE_THRES = 220
UI_SCALE = 9
UI_ARROW_SCALE = 3
TACTILE_Y_START = 0


# ==================== Data Processing Functions ====================
def split_left_right_from_flat(flat_data):
    img_raw = np.array(flat_data, dtype=np.float32).reshape(TACTILE_SHAPE, order='F')

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

    if TACTILE_Y_START > 0:
        y_start_half = TACTILE_Y_START // 2
        fx_Lf[:y_start_half, :] = 0
        fy_Lf[:y_start_half, :] = 0
        fx_Rf[:y_start_half, :] = 0
        fy_Rf[:y_start_half, :] = 0

    return fx_Lf, fy_Lf, fx_Rf, fy_Rf


def get_pressure_display(patch_img, scale=9):
    norm = np.clip(patch_img / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    color = (color * 0.75).astype(np.uint8)
    disp = cv2.resize(color, (patch_img.shape[1] * scale, patch_img.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)
    return disp


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
            if total[i, j] < TACTILE_THRES * 1.5:
                continue

            fx, fy = fx_mat[i, j], fy_mat[i, j]
            if np.sqrt(fx ** 2 + fy ** 2) < 0.001:
                continue

            cx = int((0.5 + 2 * j) * scale)
            cy = int((0.5 + 2 * i) * scale)
            dx = int(fx * arrow_scale * scale)
            dy = int(fy * arrow_scale * scale)

            if abs(dx) > 1 or abs(dy) > 1:
                arrow_len = np.sqrt(dx ** 2 + dy ** 2)
                tip_len = min(8.0 / arrow_len, 0.4) if arrow_len > 1 else 0.3
                cv2.arrowedLine(disp, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2, tipLength=tip_len)

    return disp


# ==================== Main Visualizer Player Class ====================
class TactileDataPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Tactile Sensor Data Visualizer")
        self.root.geometry("1000x750")

        self.data = None
        self.tactile_frames = []
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0

        self.build_ui()

    def build_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_load = tk.Button(control_frame, text="📂 Load JSON", command=self.browse_file,
                                  font=("Arial", 10, "bold"))
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_filepath = tk.Label(control_frame, text="No file loaded", fg="gray", font=("Arial", 9))
        self.lbl_filepath.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        display_frame = tk.Frame(self.root, bg="black")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_label = tk.Label(display_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_time = tk.Label(progress_frame, text="00:00.000", font=("Consolas", 10))
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_slider = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                         variable=self.progress_var, command=self.on_progress_change)
        self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.lbl_total_time = tk.Label(progress_frame, text="00:00.000", font=("Consolas", 10))
        self.lbl_total_time.pack(side=tk.LEFT, padx=5)

        self.lbl_frame = tk.Label(progress_frame, text="Frame: 0 / 0", font=("Consolas", 9), fg="gray")
        self.lbl_frame.pack(side=tk.LEFT, padx=10)

        playback_frame = tk.Frame(self.root)
        playback_frame.pack(fill=tk.X, padx=10, pady=10)

        self.btn_prev = tk.Button(playback_frame, text="⏮ Prev", command=self.prev_frame,
                                  font=("Arial", 10), width=10)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(playback_frame, text="▶ Play", command=self.toggle_play,
                                  font=("Arial", 10, "bold"), width=10, bg="#d4edda")
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(playback_frame, text="Next ⏭", command=self.next_frame,
                                  font=("Arial", 10), width=10)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        tk.Label(playback_frame, text="  Speed:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(20, 5))

        self.speed_var = tk.StringVar(value="1.0x")
        speed_options = ["0.1x", "0.25x", "0.5x", "1.0x", "2.0x", "4.0x", "10.0x"]
        self.speed_combo = ttk.Combobox(playback_frame, textvariable=self.speed_var,
                                        values=speed_options, width=8, state="readonly")
        self.speed_combo.pack(side=tk.LEFT, padx=5)
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        stats_frame = tk.LabelFrame(self.root, text="Statistics", font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_stats = tk.Label(stats_frame, text="Load a file to view statistics", font=("Consolas", 9), fg="gray")
        self.lbl_stats.pack(pady=5)

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Tactile Sensor JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath):
        try:
            self.lbl_filepath.config(text=f"Loading: {filepath}", fg="orange")
            self.root.update()

            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            if 'tactile' not in self.data or len(self.data['tactile']) == 0:
                messagebox.showerror("Error", "No tactile data found in JSON file!")
                return

            self.tactile_frames = self.data['tactile']
            self.total_frames = len(self.tactile_frames)
            self.current_frame = 0

            self.lbl_filepath.config(text=filepath, fg="green")
            self.progress_slider.config(to=self.total_frames - 1)

            if self.total_frames > 0:
                total_time = self.tactile_frames[-1]['t']
                self.lbl_total_time.config(text=self.format_time(total_time))

            self.update_stats()
            self.display_frame(0)

            print(f"✅ Successfully loaded {self.total_frames} frames")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
            self.lbl_filepath.config(text=f"Failed to load: {e}", fg="red")

    def update_stats(self):
        if not self.tactile_frames:
            return

        total_time = self.tactile_frames[-1]['t']
        avg_fps = self.total_frames / total_time if total_time > 0 else 0

        sample = self.tactile_frames[0]
        has_shear = 'fx' in sample and 'fy' in sample

        stats_text = (f"Total Frames: {self.total_frames} | "
                      f"Total Time: {total_time:.2f}s | "
                      f"Avg FPS: {avg_fps:.1f} Hz | "
                      f"Shear Data: {'✅ Yes' if has_shear else '❌ No'}")

        self.lbl_stats.config(text=stats_text, fg="black")

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def display_frame(self, frame_idx):
        if not self.tactile_frames or frame_idx < 0 or frame_idx >= self.total_frames:
            return

        frame_data = self.tactile_frames[frame_idx]

        current_time = frame_data['t']
        self.lbl_time.config(text=self.format_time(current_time))
        self.lbl_frame.config(text=f"Frame: {frame_idx + 1} / {self.total_frames}")
        self.progress_var.set(frame_idx)

        flat_v = frame_data['v']
        left, right = split_left_right_from_flat(flat_v)

        target_w = 8 * UI_SCALE
        target_h = 64 * UI_SCALE

        l_pressure = get_pressure_display(left, UI_SCALE)
        r_pressure = get_pressure_display(right, UI_SCALE)

        if 'fx' in frame_data and 'fy' in frame_data:
            fx_left, fy_left, fx_right, fy_right = split_left_right_quiver(frame_data['fx'], frame_data['fy'])

            l_shear = get_quiver_display_from_precomputed(left, fx_left, fy_left, UI_SCALE, UI_ARROW_SCALE)
            r_shear = get_quiver_display_from_precomputed(right, fx_right, fy_right, UI_SCALE, UI_ARROW_SCALE)
        else:
            l_shear = self.calculate_shear_display(left)
            r_shear = self.calculate_shear_display(right)

        cv2.putText(l_pressure, f"L:{np.max(left):.0f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(r_pressure, f"R:{np.max(right):.0f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(l_shear, "Shear L", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(r_shear, "Shear R", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        sep = np.ones((target_h, 10, 3), dtype=np.uint8) * 50
        group_sep = np.ones((target_h, 30, 3), dtype=np.uint8) * 100

        combined = np.hstack([l_pressure, sep, r_pressure, group_sep, l_shear, sep, r_shear])

        rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=pil_image)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def calculate_shear_display(self, patch_img):
        H, W = patch_img.shape
        norm = np.clip(patch_img / TACTILE_MAX_FORCE * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
        color = (color * 0.75).astype(np.uint8)
        disp = cv2.resize(color, (W * UI_SCALE, H * UI_SCALE), interpolation=cv2.INTER_NEAREST)

        blocks = patch_img.reshape(H // 2, 2, W // 2, 2)
        left = blocks[:, :, :, 0].sum(axis=1)
        right = blocks[:, :, :, 1].sum(axis=1)
        top = blocks[:, 0, :, :].sum(axis=2)
        bottom = blocks[:, 1, :, :].sum(axis=2)

        total = blocks.sum(axis=(1, 3))
        denom = 4 * 1023.0 - total
        denom[np.abs(denom) < 1e-6] = 1e-6

        Fx = (right - left) / denom
        Fy = (bottom - top) / denom

        for i in range(H // 2):
            for j in range(W // 2):
                if total[i, j] < TACTILE_THRES * 1.5: continue
                fx, fy = Fx[i, j], Fy[i, j]
                if np.sqrt(fx ** 2 + fy ** 2) < 0.001: continue

                cx = int((0.5 + 2 * j) * UI_SCALE)
                cy = int((0.5 + 2 * i) * UI_SCALE)
                dx = int(fx * UI_ARROW_SCALE * UI_SCALE)
                dy = int(fy * UI_ARROW_SCALE * UI_SCALE)
                if abs(dx) > 1 or abs(dy) > 1:
                    cv2.arrowedLine(disp, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2, tipLength=0.3)
        return disp

    def on_progress_change(self, value):
        if not self.tactile_frames: return
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self.current_frame = frame_idx
            self.display_frame(frame_idx)

    def on_speed_change(self, event=None):
        speed_str = self.speed_var.get()
        self.playback_speed = float(speed_str.replace('x', ''))

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Play", bg="#d4edda")
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸ Pause", bg="#f8d7da")
            threading.Thread(target=self.playback_loop, daemon=True).start()

    def playback_loop(self):
        last_time = time.time()
        while self.is_playing and self.current_frame < self.total_frames - 1:
            current_t = self.tactile_frames[self.current_frame]['t']
            next_t = self.tactile_frames[self.current_frame + 1]['t']
            wait_time = (next_t - current_t) / self.playback_speed

            elapsed = time.time() - last_time
            if elapsed < wait_time:
                time.sleep(max(0, wait_time - elapsed))

            last_time = time.time()
            self.current_frame += 1
            self.root.after(0, lambda: self.display_frame(self.current_frame))

        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.root.after(0, lambda: self.btn_play.config(text="▶ Play", bg="#d4edda"))

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame(self.current_frame)

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame(self.current_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = TactileDataPlayer(root)
    root.mainloop()