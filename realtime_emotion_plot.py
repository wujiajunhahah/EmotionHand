#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 实时EMG+GSR可视化系统
整合实时串口读取、特征提取、状态识别和可视化演示

基于用户提供的专业实时脚本，优化整合到EmotionHand项目中
"""

import time
import sys
import glob
import csv
import numpy as np
from collections import deque
from scipy.signal import welch, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import threading
import os

# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 串口读取（自动发现） ----------
def auto_port(pattern="/dev/tty.usbmodem*"):
    """自动发现串口设备"""
    c = sorted(glob.glob(pattern))
    if not c:
        print("⚠️ 未找到串口，切换到模拟数据模式。")
        return None
    print("✅ 使用串口：", c[0])
    return c[0]

class SerialReader:
    """串口数据读取器"""
    def __init__(self, port=None, baud=115200):
        self.mock = port is None
        if not self.mock:
            import serial
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.02)
        self.t = 0.0

    def read(self):
        if self.mock:
            # 模拟数据：立即看到实时曲线
            self.t += 0.02
            emg = 300 + 100*np.sin(2*np.pi*2*self.t) + 80*np.random.randn()
            gsr = 600 + 80*np.sin(2*np.pi*0.2*self.t) + 20*np.random.randn()
            return emg, gsr, time.time()

        line = self.ser.readline().decode(errors="ignore").strip()
        # 期望 Arduino 输出: "emg,gsr"
        if "," in line:
            try:
                a,b = line.split(",")[:2]
                return float(a), float(b), time.time()
            except:
                return None
        return None

# ---------- 信号处理与特征 ----------
def bandpass_emg(x, fs=1000, low=20, high=450):
    """EMG信号带通滤波"""
    b,a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def median_frequency(x, fs=1000):
    """计算中值频率"""
    if len(x) < 128:
        return 0.0
    f, Pxx = welch(x, fs=fs, nperseg=256, noverlap=128)
    if Pxx.sum() <= 0:
        return 0.0
    c = np.cumsum(Pxx)/np.sum(Pxx)
    return float(f[np.searchsorted(c, 0.5)])

class Calibrator:
    """个性化校准器"""
    def __init__(self):
        self.e, self.g = [], []
        self.ready = False

    def feed(self, emg, gsr):
        if not self.ready:
            self.e.append(emg)
            self.g.append(gsr)
            if len(self.e) >= 3000:  # ~60s @ 50Hz帧
                self.e_p10, self.e_p90 = np.percentile(self.e,[10,90])
                self.g_p10, self.g_p90 = np.percentile(self.g,[10,90])
                self.ready = True
                print("✅ 校准完成！")

    def norm(self, emg, gsr):
        if not self.ready:
            return None
        def nq(x, lo, hi):
            return np.clip((x-lo)/max(1e-6, hi-lo), 0, 1)
        return float(nq(emg, self.e_p10, self.e_p90)), float(nq(gsr, self.g_p10, self.g_p90))

class StateDecider:
    """情绪状态判定器"""
    def __init__(self, alpha=0.7):
        self.hist = deque(maxlen=10)
        self.conf = 0.0
        self.alpha = alpha

    def decide(self, emg_rms01, mdf01, gsr01):
        # 简单规则（可后续替换为训练的模型概率）
        if emg_rms01 < 0.25 and gsr01 < 0.25:
            s, p = "Relaxed", 0.8
        elif 0.25 <= emg_rms01 <= 0.55 and 0.25 <= gsr01 <= 0.55 and mdf01 >= 0.5:
            s, p = "Focused", 0.75
        elif emg_rms01 > 0.55 and gsr01 > 0.55 and mdf01 >= 0.6:
            s, p = "Stressed", 0.85
        elif emg_rms01 < 0.25 and mdf01 < 0.35 and gsr01 <= 0.4:
            s, p = "Fatigued", 0.7
        else:
            s, p = "Neutral", 0.55

        self.conf = self.alpha*p + (1-self.alpha)*self.conf
        self.hist.append(s)
        # 多数投票抑抖
        s_major = max(set(self.hist), key=self.hist.count)
        return s_major, float(self.conf)

# ---------- 3D手部模型可视化 ----------
class HandModel3D:
    """简化的3D手部模型"""
    def __init__(self):
        self.palm_length = 0.85
        self.palm_width = 0.85
        self.finger_lengths = [0.65, 0.75, 0.70, 0.55]
        self.thumb_length = 0.55
        self.finger_width = 0.18

        # 情绪状态到颜色的映射
        self.state_colors = {
            'Relaxed': '#3498db',      # 蓝色
            'Focused': '#2ecc71',      # 绿色
            'Stressed': '#e74c3c',     # 红色
            'Fatigued': '#f39c12'      # 黄色
        }

    def get_simple_hand_position(self, state):
        """根据情绪状态返回简化的手部位置"""
        # 不同情绪状态下的手势特征
        gestures = {
            'Relaxed': 'Open',
            'Focused': 'Pinch',
            'Stressed': 'Fist',
            'Fatigued': 'Neutral'
        }
        return gestures.get(state, 'Neutral')

    def draw_simple_hand(self, ax, state, confidence):
        """绘制简化的手部表示"""
        color = self.state_colors.get(state, '#95a5a6')
        gesture = self.get_simple_hand_position(state)

        # 清除之前的图形
        ax.clear()

        # 根据不同手势绘制简化表示
        if gesture == 'Open':
            # 张开的手
            theta = np.linspace(0, 2*np.pi, 5)
            x = 0.3 * np.cos(theta)
            y = 0.3 * np.sin(theta)
            ax.fill(x, y, color=color, alpha=0.7)
            ax.set_title(f'✋ Open Hand\n{state} (置信度: {confidence:.2f})')
        elif gesture == 'Fist':
            # 拳头
            circle = plt.Circle((0, 0), 0.2, color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.set_title(f'✊ Fist\n{state} (置信度: {confidence:.2f})')
        elif gesture == 'Pinch':
            # 捏合手势
            ax.plot([0, 0.1], [0, 0.1], 'o-', color=color, linewidth=8, markersize=12)
            ax.plot([0, -0.1], [0, 0.1], 'o-', color=color, linewidth=8, markersize=12)
            ax.set_title(f'✌️ Pinch\n{state} (置信度: {confidence:.2f})')
        else:
            # 中性手势
            rect = plt.Rectangle((-0.15, -0.2), 0.3, 0.4, color=color, alpha=0.7)
            ax.add_patch(rect)
            ax.set_title(f'🖐️ Neutral\n{state} (置信度: {confidence:.2f})')

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

# ---------- 主程序 ----------
def main():
    """主程序"""
    print("🎭 EmotionHand 实时EMG+GSR可视化系统")
    print("=" * 60)
    print("📋 功能说明:")
    print("  • 实时串口读取EMG+GSR数据")
    print("  • 自动校准和归一化处理")
    print("  • 特征提取：RMS、MDF、GSR")
    print("  • 情绪状态实时识别")
    print("  • 3D可视化手势展示")
    print("  • 按s录制，按q退出")
    print("=" * 60)

    # 初始化组件
    port = auto_port()
    reader = SerialReader(port)
    calib = Calibrator()
    decide = StateDecider(alpha=0.7)
    hand_model = HandModel3D()

    # 可视化准备
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('🎭 EmotionHand - Real-time EMG+GSR Emotion Recognition',
                 fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 2, 1])

    # 子图布局
    ax_emg = fig.add_subplot(gs[0, :2])
    ax_gsr = fig.add_subplot(gs[1, :2])
    ax_features = fig.add_subplot(gs[2, :2])
    ax_hand = fig.add_subplot(gs[:2, 2])
    ax_status = fig.add_subplot(gs[2, 2])

    # 滑窗缓存
    fs_plot = 50   # 50帧/秒可视化节拍
    dur = 10       # 显示最近10秒
    nbuf = fs_plot * dur
    emg_buf = deque(maxlen=nbuf)
    gsr_buf = deque(maxlen=nbuf)
    t_buf = deque(maxlen=nbuf)

    # 特征缓存
    feature_buf = deque(maxlen=100)

    # 曲线对象
    emg_line, = ax_emg.plot([], [], lw=1.2, color="#1f77b4", label="EMG Raw")
    gsr_line, = ax_gsr.plot([], [], lw=1.2, color="#2ca02c", label="GSR Raw")

    # 设置坐标轴
    ax_emg.set_title("EMG Signal (Raw)", fontweight='bold')
    ax_emg.set_ylabel("ADC Value")
    ax_emg.set_xlim(0, dur)
    ax_emg.set_ylim(0, 1023)
    ax_emg.legend(loc='upper right')
    ax_emg.grid(True, alpha=0.3)

    ax_gsr.set_title("GSR Signal (Raw)", fontweight='bold')
    ax_gsr.set_ylabel("ADC Value")
    ax_gsr.set_xlabel("Time (s)")
    ax_gsr.set_xlim(0, dur)
    ax_gsr.set_ylim(0, 1023)
    ax_gsr.legend(loc='upper right')
    ax_gsr.grid(True, alpha=0.3)

    ax_features.set_title("Feature Analysis", fontweight='bold')
    ax_features.set_xlabel("Time (s)")
    ax_features.set_ylabel("Normalized Value")
    ax_features.set_xlim(0, 20)
    ax_features.set_ylim(0, 1)
    ax_features.grid(True, alpha=0.3)

    # 手部可视化
    ax_hand.set_title("Hand Gesture", fontweight='bold')

    # 状态面板
    ax_status.set_title("System Status", fontweight='bold')
    ax_status.axis('off')

    # 录制状态
    recording = {"on": False, "rows": []}

    last_plot = time.time()
    emg_proc_buf = deque(maxlen=256)  # 频域用

    def on_key(event):
        """键盘事件处理"""
        if event.key == 's':
            recording["on"] = not recording["on"]
            print("🎙️ 录制：", "开始" if recording["on"] else "停止")
        elif event.key == 'q':
            plt.close(fig)
            print("👋 程序退出")

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update_frame():
        """更新一帧数据"""
        nonlocal last_plot

        while plt.fignum_exists(fig.number):
            pkt = reader.read()
            if pkt is None:
                continue

            emg, gsr, ts = pkt

            # 可视化节拍控制
            now = time.time()
            if now - last_plot < 1/fs_plot:
                continue
            last_plot = now

            # 校准期
            if not calib.ready:
                calib.feed(emg, gsr)
                rem = max(0, 60 - int(len(calib.e)/50))

                # 更新校准显示
                ax_hand.clear()
                ax_hand.text(0.5, 0.5, f"Calibrating...\n{rem}s",
                           ha='center', va='center', fontsize=16, fontweight='bold')
                ax_hand.set_xlim(0, 1)
                ax_hand.set_ylim(0, 1)
                ax_hand.axis('off')

                # 更新状态
                ax_status.clear()
                ax_status.text(0.5, 0.7, "校准中", ha='center', va='center',
                             fontsize=12, fontweight='bold', color='orange')
                ax_status.text(0.5, 0.3, "请保持静息 + 轻握", ha='center', va='center',
                             fontsize=10)
                ax_status.set_xlim(0, 1)
                ax_status.set_ylim(0, 1)
                ax_status.axis('off')

                # 更新曲线
                t_buf.append(now)
                emg_buf.append(emg)
                gsr_buf.append(gsr)

                if len(t_buf) > 1:
                    xs = np.array([t - t_buf[0] for t in t_buf])
                    emg_line.set_data(xs, list(emg_buf))
                    gsr_line.set_data(xs, list(gsr_buf))
                    ax_emg.set_xlim(0, max(3, xs.max()))
                    ax_gsr.set_xlim(0, max(3, xs.max()))

                plt.pause(0.001)
                continue

            # 归一化 + 特征提取
            emg01, gsr01 = calib.norm(emg, gsr)
            emg_proc_buf.append(emg - np.mean(list(emg_buf)[-50:]) if len(emg_buf) > 50 else emg)
            emg_arr = np.array(emg_proc_buf, dtype=float)

            # EMG预处理
            if len(emg_arr) >= 128:
                emg_f = bandpass_emg(emg_arr, fs=1000)
            else:
                emg_f = emg_arr

            emg_rms = float(np.sqrt(np.mean((emg_f - emg_f.mean())**2)))
            mdf = median_frequency(emg_f, fs=1000)

            # 归一化特征
            emg_rms01 = np.clip(emg_rms/400.0, 0, 1)
            mdf01 = np.clip((mdf-40)/80.0, 0, 1)

            # 状态判定
            state, conf = decide.decide(emg_rms01, mdf01, gsr01)

            # 录制数据
            if recording["on"]:
                recording["rows"].append([ts, emg, gsr, emg_rms, mdf,
                                        emg01, gsr01, emg_rms01, mdf01, state, conf])

            # 更新数据缓存
            t_buf.append(now)
            emg_buf.append(emg)
            gsr_buf.append(gsr)
            feature_buf.append((now, emg_rms01, mdf01, gsr01))

            # 更新信号曲线
            if len(t_buf) > 1:
                xs = np.array([t - t_buf[0] for t in t_buf])
                emg_line.set_data(xs, list(emg_buf))
                gsr_line.set_data(xs, list(gsr_buf))
                ax_emg.set_xlim(0, dur)
                ax_gsr.set_xlim(0, dur)

            # 更新特征曲线
            if len(feature_buf) > 1:
                feature_times = np.array([f[0] for f in feature_buf])
                feature_times = feature_times - feature_times[0]
                emg_rms_vals = [f[1] for f in feature_buf]
                mdf_vals = [f[2] for f in feature_buf]
                gsr_vals = [f[3] for f in feature_buf]

                ax_features.clear()
                ax_features.plot(feature_times, emg_rms_vals, 'r-', label='EMG RMS', linewidth=2)
                ax_features.plot(feature_times, mdf_vals, 'g-', label='MDF', linewidth=2)
                ax_features.plot(feature_times, gsr_vals, 'b-', label='GSR', linewidth=2)
                ax_features.set_xlim(max(0, feature_times[-1]-20), feature_times[-1]+1)
                ax_features.set_ylim(0, 1)
                ax_features.set_xlabel("Time (s)")
                ax_features.set_ylabel("Normalized Value")
                ax_features.legend(loc='upper right')
                ax_features.grid(True, alpha=0.3)

            # 更新手部可视化
            hand_model.draw_simple_hand(ax_hand, state, conf)

            # 更新状态面板
            ax_status.clear()
            color_map = {
                "Relaxed": "#3498db", "Focused": "#2ecc71",
                "Stressed": "#e74c3c", "Fatigued": "#f39c12",
                "Neutral": "#95a5a6"
            }
            c = color_map.get(state, "#95a5a6")

            ax_status.text(0.5, 0.8, state, ha='center', va='center',
                         fontsize=14, fontweight='bold', color=c)
            ax_status.text(0.5, 0.6, f"置信度: {conf:.2f}", ha='center', va='center',
                         fontsize=10)
            ax_status.text(0.5, 0.4, f"EMG_RMS: {emg_rms01:.2f}", ha='center', va='center',
                         fontsize=9)
            ax_status.text(0.5, 0.3, f"MDF: {mdf01:.2f}", ha='center', va='center',
                         fontsize=9)
            ax_status.text(0.5, 0.2, f"GSR: {gsr01:.2f}", ha='center', va='center',
                         fontsize=9)

            if recording["on"]:
                ax_status.text(0.5, 0.05, "🔴 录制中", ha='center', va='center',
                             fontsize=10, color='red')

            ax_status.set_xlim(0, 1)
            ax_status.set_ylim(0, 1)
            ax_status.axis('off')

            plt.pause(0.001)

    # 启动更新线程
    update_thread = threading.Thread(target=update_frame, daemon=True)
    update_thread.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")

    # 保存录制数据
    if recording["rows"]:
        timestamp = time.strftime("runs/emotion_stream_%Y%m%d_%H%M%S.csv")
        os.makedirs("runs", exist_ok=True)
        with open(timestamp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emg_raw", "gsr_raw", "emg_rms", "mdf",
                           "emg_norm", "gsr_norm", "emg_rms_norm", "mdf_norm",
                           "state", "confidence"])
            writer.writerows(recording["rows"])
        print(f"💾 数据已保存到: {timestamp}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()