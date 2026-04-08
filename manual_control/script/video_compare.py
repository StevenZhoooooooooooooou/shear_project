#!/usr/bin/env python3
"""
慢动作视频生成器 - 精简版
仅保留接触过程，超慢速回放
"""

import os
import json
import numpy as np
import cv2
from datetime import datetime

# ================= 配置区域 =================
SESSION_DIR = r"/shear_tactile_project/manual_control/data/ball_catching/session_20260324_191424"

OUTPUT_FILENAME = "contact_slow_motion.mp4"

# 慢放倍数（越大越慢）
SLOW_MOTION_FACTOR = 50  # 原来是20，现在改为50倍慢放

# 接触检测参数
CONTACT_THRESHOLD_RATIO = 0.10  # 力超过峰值的10%即视为接触开始
PADDING_BEFORE_SEC = 0.3  # 接触前多保留0.3秒
PADDING_AFTER_SEC = 0.5  # 接触后多保留0.5秒

# 可视化参数
TACTILE_DISPLAY_SIZE = 500
ARROW_LENGTH_SCALE = 2.5


# ============================================
def generate_contact_video(session_dir):
    """生成仅包含接触过程的慢动作视频"""

    json_path = os.path.join(session_dir, "sensor_data.json")
    video_path = os.path.join(session_dir, "camera_view.avi")
    out_path = os.path.join(session_dir, OUTPUT_FILENAME)

    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print("❌ 找不到数据文件！")
        return

    print("=" * 60)
    print("🎬 接触过程慢动作生成器")
    print("=" * 60)

    # ========== 加载数据 ==========
    print("\n⏳ 加载数据...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    tactile_data = data['tactile']
    print(f"   触觉帧数: {len(tactile_data)}")

    # 解析触觉数据
    tac_times = []
    frames_fx = []
    frames_fy = []

    for frame in tactile_data:
        tac_times.append(frame['t'])
        fx = np.array(frame['fx']).reshape((16, 16), order='F')
        fy = np.array(frame['fy']).reshape((16, 16), order='F')
        frames_fx.append(fx)
        frames_fy.append(fy)

    tac_times = np.array(tac_times)
    frames_fx = np.array(frames_fx)
    frames_fy = np.array(frames_fy)

    # 基线校准
    base_fx = np.mean(frames_fx[:30], axis=0)
    base_fy = np.mean(frames_fy[:30], axis=0)
    frames_fx = frames_fx - base_fx
    frames_fy = frames_fy - base_fy

    # 计算力大小
    all_mags = np.sqrt(frames_fx ** 2 + frames_fy ** 2)
    frame_max_mags = all_mags.max(axis=(1, 2))

    global_max = np.percentile(all_mags, 99.5)
    if global_max < 0.01:
        global_max = 0.1

    # ========== 检测接触时段 ==========
    print("\n⏳ 检测接触时段...")

    baseline_mag = np.mean(frame_max_mags[:30])
    peak_mag = frame_max_mags.max()

    # 动态阈值
    threshold = baseline_mag + (peak_mag - baseline_mag) * CONTACT_THRESHOLD_RATIO

    # 找到接触起止帧
    contact_mask = frame_max_mags > threshold
    contact_indices = np.where(contact_mask)[0]

    if len(contact_indices) == 0:
        print("⚠️ 未检测到明显接触，将处理全部数据")
        tac_start_idx = 0
        tac_end_idx = len(tactile_data) - 1
    else:
        tac_start_idx = contact_indices[0]
        tac_end_idx = contact_indices[-1]

    # 添加前后padding
    tac_fps = len(tactile_data) / (tac_times[-1] - tac_times[0])
    pad_before = int(PADDING_BEFORE_SEC * tac_fps)
    pad_after = int(PADDING_AFTER_SEC * tac_fps)

    tac_start_idx = max(0, tac_start_idx - pad_before)
    tac_end_idx = min(len(tactile_data) - 1, tac_end_idx + pad_after)

    contact_start_time = tac_times[tac_start_idx]
    contact_end_time = tac_times[tac_end_idx]
    contact_duration = contact_end_time - contact_start_time

    print(f"   基线力值: {baseline_mag:.4f}")
    print(f"   峰值力: {peak_mag:.4f}")
    print(f"   检测阈值: {threshold:.4f}")
    print(f"   接触时段: {contact_start_time:.3f}s ~ {contact_end_time:.3f}s")
    print(f"   接触持续: {contact_duration:.3f}s ({tac_end_idx - tac_start_idx + 1} 帧)")

    # ========== 打开视频 ==========
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n   视频: {orig_w}x{orig_h} @ {video_fps:.1f}fps, {video_frames}帧")

    # 计算视频对应的起止帧（按比例映射）
    video_start_frame = int(tac_start_idx / (len(tactile_data) - 1) * (video_frames - 1))
    video_end_frame = int(tac_end_idx / (len(tactile_data) - 1) * (video_frames - 1))

    print(f"   对应视频帧: {video_start_frame} ~ {video_end_frame}")

    # ========== 设置输出 ==========
    tac_size = TACTILE_DISPLAY_SIZE
    final_w = orig_w + tac_size
    final_h = max(orig_h, tac_size + 150)

    output_fps = video_fps / SLOW_MOTION_FACTOR
    output_frames = video_end_frame - video_start_frame + 1
    output_duration = output_frames / output_fps

    print(f"\n🎬 输出设置:")
    print(f"   分辨率: {final_w}x{final_h}")
    print(f"   帧率: {output_fps:.2f} FPS ({SLOW_MOTION_FACTOR}x 慢放)")
    print(f"   输出帧数: {output_frames}")
    print(f"   输出时长: {output_duration:.1f}s")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, output_fps, (final_w, final_h))

    # ========== 跳转到起始帧 ==========
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    # ========== 逐帧处理 ==========
    print(f"\n⏳ 渲染中...")
    start_time = datetime.now()

    for frame_idx in range(video_start_frame, video_end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # 按比例映射到触觉帧
        ratio = (frame_idx - video_start_frame) / max(video_end_frame - video_start_frame, 1)
        tac_idx = tac_start_idx + int(ratio * (tac_end_idx - tac_start_idx))
        tac_idx = np.clip(tac_idx, 0, len(tactile_data) - 1)

        # 当前力场
        fx = frames_fx[tac_idx]
        fy = frames_fy[tac_idx]
        mag = all_mags[tac_idx]

        # 渲染触觉可视化
        tactile_img = render_tactile(fx, fy, mag, global_max, tac_size, final_h,
                                     tac_times[tac_idx], tac_idx, len(tactile_data),
                                     contact_start_time, frame_max_mags[tac_idx])

        # 拼接
        if orig_h < final_h:
            pad = (final_h - orig_h) // 2
            frame = cv2.copyMakeBorder(frame, pad, final_h - orig_h - pad, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

        combined = np.hstack((frame, tactile_img))
        out.write(combined)

        # 进度
        progress_idx = frame_idx - video_start_frame
        if progress_idx % 50 == 0:
            progress = progress_idx / output_frames * 100
            print(f"   {progress:.0f}% ({progress_idx}/{output_frames})")

    cap.release()
    out.release()

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"✅ 完成！")
    print(f"   耗时: {elapsed:.1f}s")
    print(f"   输出: {out_path}")
    print(f"   时长: {output_duration:.1f}s (原{contact_duration:.2f}s的{SLOW_MOTION_FACTOR}x慢放)")
    print(f"{'=' * 60}")


def render_tactile(fx, fy, mag, global_max, size, total_h, current_time,
                   tac_idx, total_tac, contact_start, current_max_mag):
    """渲染触觉可视化面板"""

    canvas = np.zeros((total_h, size, 3), dtype=np.uint8)

    cell_size = (size - 40) / 16.0
    offset_x = 20
    offset_y = 60

    # 热力图背景
    norm_mag = np.clip((mag / global_max) * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.resize(norm_mag, (int(cell_size * 16), int(cell_size * 16)),
                         interpolation=cv2.INTER_NEAREST)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap_color = (heatmap_color * 0.5).astype(np.uint8)

    hh, hw = heatmap_color.shape[:2]
    canvas[offset_y:offset_y + hh, offset_x:offset_x + hw] = heatmap_color

    # 绘制箭头
    for i in range(16):
        for j in range(16):
            f_x, f_y = fx[i, j], fy[i, j]
            f_mag = mag[i, j]

            if f_mag < global_max * 0.05:
                continue

            cx = offset_x + int((j + 0.5) * cell_size)
            cy = offset_y + int((i + 0.5) * cell_size)

            arrow_len = (f_mag / global_max) * cell_size * ARROW_LENGTH_SCALE

            if f_mag > 1e-6:
                dx = int(f_x / f_mag * arrow_len)
                dy = int(f_y / f_mag * arrow_len)

                if abs(dx) > 1 or abs(dy) > 1:
                    intensity = min(f_mag / global_max, 1.0)
                    b, g, r = 0, int(255 * (1 - intensity)), int(255 * intensity)
                    cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy + dy),
                                    (b, g, r), 2, tipLength=0.3)

    # 网格
    for i in range(17):
        pos = offset_x + int(i * cell_size)
        cv2.line(canvas, (pos, offset_y), (pos, offset_y + int(16 * cell_size)), (40, 40, 40), 1)
    for i in range(17):
        pos = offset_y + int(i * cell_size)
        cv2.line(canvas, (offset_x, pos), (offset_x + int(16 * cell_size), pos), (40, 40, 40), 1)

    # 信息面板
    # 标题
    cv2.putText(canvas, "CONTACT ANALYSIS", (size // 2 - 100, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

    info_y = offset_y + int(16 * cell_size) + 35

    # 相对时间（从接触开始算）
    rel_time = current_time - contact_start
    cv2.putText(canvas, f"T = {rel_time:+.3f} s", (20, info_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # 力统计
    net_fx = np.sum(fx)
    net_fy = np.sum(fy)

    cv2.putText(canvas, f"Max: {current_max_mag:.4f}", (20, info_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 255, 100), 1)
    cv2.putText(canvas, f"Fx: {net_fx:+.2f}  Fy: {net_fy:+.2f}", (20, info_y + 65),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 180, 180), 1)

    # 力值条形图
    bar_x = size - 60
    bar_max_h = 180
    bar_h = int((current_max_mag / global_max) * bar_max_h)
    bar_h = min(bar_h, bar_max_h)

    bar_top = info_y - bar_max_h - 10
    cv2.rectangle(canvas, (bar_x, bar_top + bar_max_h - bar_h), (bar_x + 35, bar_top + bar_max_h), (0, 200, 0), -1)
    cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + 35, bar_top + bar_max_h), (100, 100, 100), 2)
    cv2.putText(canvas, "Force", (bar_x - 5, bar_top + bar_max_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # 帧号
    cv2.putText(canvas, f"#{tac_idx}", (size - 80, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return canvas


if __name__ == "__main__":
    generate_contact_video(SESSION_DIR)