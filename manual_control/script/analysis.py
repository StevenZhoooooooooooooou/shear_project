import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================
DATA_FILE = r"D:\Python_Project\shear_tactile_project\manual_control\data\ball_catching\session_20260324_191424\sensor_data.json"


# ============================================

def analyze_3d_forces(filepath):
    print("⏳ 正在加载并提取 3D 触觉受力数据...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    tactile_data = data['tactile']
    aperture_data = data.get('aperture', [])

    times = []
    net_fx = []
    net_fy = []
    net_fz = []  # 新增：向内的法向力 (Z轴)

    # 1. 提取三维力分量
    for frame in tactile_data:
        times.append(frame['t'])

        # 剪切力 16x16 矩阵 (X 和 Y 方向)
        fx_mat = np.array(frame['fx']).reshape((16, 16), order='F')
        fy_mat = np.array(frame['fy']).reshape((16, 16), order='F')

        # 原始法向压力 32x32 矩阵 (Z 方向，即向内部的挤压力)
        fz_mat = np.array(frame['v']).reshape((32, 32), order='F')

        net_fx.append(np.sum(fx_mat))
        net_fy.append(np.sum(fy_mat))
        net_fz.append(np.sum(fz_mat))

    times = np.array(times)

    # 简单平滑处理，滤除高频电噪声
    window = 5
    smooth_fx = np.convolve(np.array(net_fx), np.ones(window) / window, mode='same')
    smooth_fy = np.convolve(np.array(net_fy), np.ones(window) / window, mode='same')
    smooth_fz = np.convolve(np.array(net_fz), np.ones(window) / window, mode='same')

    # 计算剪切合力大小
    magnitude = np.sqrt(smooth_fx ** 2 + smooth_fy ** 2)

    # 提取视觉开合度数据
    apt_times = [frame['t'] for frame in aperture_data if frame['dist_cm'] is not None]
    apt_dists = [frame['dist_cm'] for frame in aperture_data if frame['dist_cm'] is not None]

    # ================= 动态底噪校准与精准触发 =================
    # 使用向内的法向力 (Fz) 来判定接触瞬间是最准确的物理逻辑
    baseline_fz = np.mean(smooth_fz[:30])
    peak_fz = np.max(smooth_fz)

    # 动态阈值：底噪之上，法向力突变增幅的 5% 处作为初始接触瞬间
    trigger_threshold = baseline_fz + (peak_fz - baseline_fz) * 0.05
    trigger_idx = np.argmax(smooth_fz > trigger_threshold)
    # ==========================================================

    # ================= 开始高质量绘图 =================
    plt.style.use('seaborn-v0_8-paper')
    # 使用 3 个上下堆叠的子图，共享时间轴
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 共同的接触瞬间基准线
    contact_time = times[trigger_idx]

    # --- 图 1：Z轴 (向内法向力) 与 夹爪开合度 ---
    color_z = '#9467bd'  # 紫色
    ax1.plot(times, smooth_fz, color=color_z, linewidth=2.5, label='Normal Force $F_z$ (Inward Pressure)')
    ax1.set_ylabel("Normal Force Proxy", color=color_z, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_z)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 绘制开合度 (右侧 Y 轴)
    if len(apt_times) > 0:
        ax1_twin = ax1.twinx()
        color_apt = '#ff7f0e'  # 橙色
        ax1_twin.plot(apt_times, apt_dists, color=color_apt, linewidth=2.5, linestyle='-.', label='Aperture Width')
        ax1_twin.set_ylabel("Aperture (cm)", color=color_apt, fontsize=12, fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor=color_apt)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

    ax1.axvline(x=contact_time, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title("3D Force Components", fontsize=16, fontweight='bold')

    # --- 图 2：X轴与Y轴 (剪切力分量) ---
    # ⚠️ 修改点：互换了 smooth_fy 和 smooth_fx，并精简了 label
    ax2.plot(times, smooth_fy, color='#d62728', linewidth=2, label='Horizontal Shear $F_x$')
    ax2.plot(times, smooth_fx, color='#1f77b4', linewidth=2, label='Vertical Shear $F_y$')
    ax2.set_ylabel("Directional Shear Proxy", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.axvline(x=contact_time, color='gray', linestyle='--', alpha=0.7)

    # --- 图 3：XY平面合力大小 (Shear Magnitude) ---
    color_mag = '#2ca02c'  # 绿色
    ax3.plot(times, magnitude, color=color_mag, linewidth=2.5, label='Total Shear Magnitude $|F_{xy}|$')
    ax3.set_ylabel("Shear Magnitude", color=color_mag, fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color_mag)
    ax3.set_xlabel("Time (s)", fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', fontsize=11)

    ax3.axvline(x=contact_time, color='gray', linestyle='--', alpha=0.7)

    # 在最底部的图上标注接触线说明
    ax3.text(contact_time + 0.05, np.max(magnitude) * 0.5, 'Initial Contact', rotation=90, verticalalignment='center',
             fontsize=11, fontweight='bold', color='dimgray')

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(filepath), "3d_force_components_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已生成并保存至:\n{save_path}")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        analyze_3d_forces(DATA_FILE)
    else:
        print(f"❌ 找不到文件: {DATA_FILE}")