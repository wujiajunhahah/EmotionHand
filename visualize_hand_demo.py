#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 手部模型可视化演示
用假数据模拟EMG+GSR信号识别和Unity 3D可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import threading
import queue
from dataclasses import dataclass
from typing import Tuple, List, Dict
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class EmotionData:
    """情绪数据结构"""
    gesture: str
    state: str
    confidence: float
    emg_signal: np.ndarray
    gsr_signal: float
    timestamp: float

class HandModel3D:
    """3D手部模型"""

    def __init__(self):
        # 手部几何参数 (相对单位)
        self.palm_length = 0.85
        self.palm_width = 0.85
        self.finger_lengths = [0.65, 0.75, 0.70, 0.55]  # 食指到小指
        self.thumb_length = 0.55
        self.finger_width = 0.18

        # 关节参数
        self.joint_bend_max = [90, 80, 70, 60]  # 各手指最大弯曲角度
        self.joint_positions = []
        self.gesture_bends = {
            'Fist': [85, 80, 75, 70],
            'Open': [5, 5, 5, 5],
            'Pinch': [10, 75, 80, 85],
            'Point': [10, 10, 10, 80],
            'Peace': [10, 10, 10, 10],
            'Neutral': [20, 20, 20, 20]
        }

    def get_finger_joints(self, gesture: str, finger_idx: int) -> List[Tuple[float, float, float]]:
        """计算手指关节位置"""
        bend_angles = self.gesture_bends.get(gesture, [20, 20, 20, 20])
        bend_angle = bend_angles[min(finger_idx, 3)]

        # 手指根部位置 (在手掌上)
        if finger_idx == 0:  # 拇指
            base_x, base_y, base_z = -self.palm_width/2, 0, 0
        else:  # 其他手指
            finger_spacing = self.palm_width / 5
            base_x = -self.palm_width/2 + finger_spacing * finger_idx
            base_y, base_z = self.palm_length, 0

        joints = [(base_x, base_y, base_z)]

        # 计算弯曲后的关节位置
        length = self.finger_lengths[min(finger_idx, 3)]
        segments = 3
        segment_length = length / segments

        current_x, current_y, current_z = base_x, base_y, base_z

        for i in range(segments):
            # 弯曲效果
            bend_rad = np.radians(bend_angle * (i + 1) / segments)
            current_x += segment_length * np.sin(bend_rad) * 0.3
            current_y += segment_length * np.cos(bend_rad)
            current_z += segment_length * np.sin(bend_rad) * 0.2 * (1 if i % 2 == 0 else -1)
            joints.append((current_x, current_y, current_z))

        return joints

class SignalSimulator:
    """信号模拟器"""

    def __init__(self):
        self.sample_rate = 1000
        self.gestures = ['Fist', 'Open', 'Pinch', 'Point', 'Peace', 'Neutral']
        self.states = ['Relaxed', 'Focused', 'Stressed', 'Fatigued']
        self.current_gesture = 'Neutral'
        self.current_state = 'Relaxed'
        self.time = 0

    def generate_emg_signal(self, duration: float, gesture: str) -> np.ndarray:
        """生成EMG信号"""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # 基础频率 (肌肉活动)
        gesture_frequencies = {
            'Fist': [30, 50, 80, 120, 200],
            'Open': [10, 25, 40, 60, 90],
            'Pinch': [40, 70, 110, 180, 250],
            'Point': [20, 45, 85, 150, 220],
            'Peace': [15, 35, 65, 110, 180],
            'Neutral': [5, 15, 30, 45, 80]
        }

        freqs = gesture_frequencies.get(gesture, [10, 25, 40, 60, 90])
        signal = np.zeros(n_samples)

        # 8通道EMG信号
        channels = []
        for ch in range(8):
            channel_signal = 0
            for i, freq in enumerate(freqs):
                amplitude = 0.3 / (i + 1)  # 递减幅度
                phase = np.random.random() * 2 * np.pi
                channel_signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

            # 添加噪声
            channel_signal += 0.1 * np.random.randn(n_samples)

            # 手势相关的调制
            if gesture != 'Neutral':
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
                channel_signal *= envelope

            channels.append(channel_signal)

        return np.array(channels).T

    def generate_gsr_signal(self, duration: float, state: str) -> float:
        """生成GSR信号"""
        # 状态相关的GSR基线值
        state_values = {
            'Relaxed': 0.1 + 0.05 * np.sin(self.time * 0.1),
            'Focused': 0.2 + 0.08 * np.sin(self.time * 0.15),
            'Stressed': 0.4 + 0.15 * np.sin(self.time * 0.2) + 0.1 * np.random.random(),
            'Fatigued': 0.25 + 0.12 * np.sin(self.time * 0.12)
        }

        return state_values.get(state, 0.15)

    def update(self):
        """更新模拟器状态"""
        self.time += 0.1

        # 随机切换手势
        if np.random.random() < 0.05:  # 5%概率切换手势
            self.current_gesture = np.random.choice(self.gestures)

        # 随机切换状态
        if np.random.random() < 0.03:  # 3%概率切换状态
            self.current_state = np.random.choice(self.states)

class EmotionHandVisualizer:
    """EmotionHand可视化器"""

    def __init__(self):
        self.hand_model = HandModel3D()
        self.signal_simulator = SignalSimulator()
        self.data_queue = queue.Queue()
        self.current_data = None

        # 状态颜色映射
        self.state_colors = {
            'Relaxed': '#3498db',      # 蓝色
            'Focused': '#2ecc71',      # 绿色
            'Stressed': '#e74c3c',     # 红色
            'Fatigued': '#f39c12'      # 黄色
        }

        # 手势颜色
        self.gesture_colors = {
            'Fist': '#8e44ad',         # 紫色
            'Open': '#95a5a6',         # 灰色
            'Pinch': '#e67e22',        # 橙色
            'Point': '#16a085',        # 青色
            'Peace': '#27ae60',        # 绿色
            'Neutral': '#34495e'       # 深灰色
        }

    def simulate_real_time_data(self):
        """模拟实时数据流"""
        while True:
            # 更新模拟器
            self.signal_simulator.update()

            # 生成信号数据
            emg_signal = self.signal_simulator.generate_emg_signal(
                0.1, self.signal_simulator.current_gesture
            )
            gsr_signal = self.signal_simulator.generate_gsr_signal(
                0.1, self.signal_simulator.current_state
            )

            # 创建数据对象
            data = EmotionData(
                gesture=self.signal_simulator.current_gesture,
                state=self.signal_simulator.current_state,
                confidence=0.6 + 0.3 * np.random.random(),
                emg_signal=emg_signal[-1] if len(emg_signal) > 0 else np.zeros(8),
                gsr_signal=gsr_signal,
                timestamp=time.time()
            )

            # 放入队列
            if not self.data_queue.full():
                self.data_queue.put(data)

            time.sleep(0.1)  # 100ms间隔

    def create_3d_hand_plot(self, fig, position):
        """创建3D手部图"""
        ax = fig.add_subplot(2, 3, position, projection='3d')
        ax.set_title('3D Hand Model', fontsize=10, fontweight='bold')

        # 获取当前数据
        if self.current_data:
            gesture = self.current_data.gesture
            state = self.current_data.state
            confidence = self.current_data.confidence

            # 设置颜色
            hand_color = self.state_colors.get(state, '#95a5a6')
            alpha = 0.3 + 0.7 * confidence  # 透明度基于置信度
        else:
            gesture = 'Neutral'
            state = 'Relaxed'
            hand_color = '#95a5a6'
            alpha = 0.5

        # 绘制手掌
        palm_corners = [
            [-self.hand_model.palm_width/2, 0, -self.hand_model.palm_width/2],
            [self.hand_model.palm_width/2, 0, -self.hand_model.palm_width/2],
            [self.hand_model.palm_width/2, 0, self.hand_model.palm_width/2],
            [-self.hand_model.palm_width/2, 0, self.hand_model.palm_width/2]
        ]

        # 手掌顶面
        palm_top = [[p[0], p[1] + 0.1, p[2]] for p in palm_corners]
        palm_collection = Poly3DCollection([palm_top], alpha=alpha, facecolor=hand_color, edgecolor='black')
        ax.add_collection3d(palm_collection)

        # 绘制手指
        for finger_idx in range(5):
            joints = self.hand_model.get_finger_joints(gesture, finger_idx)

            # 绘制手指线条
            xs, ys, zs = zip(*joints)
            ax.plot(xs, ys, zs, 'o-', color=hand_color, linewidth=3, markersize=4, alpha=alpha)

        # 设置坐标轴
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 2])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加状态文本
        ax.text2D(0.05, 0.95, f'Gesture: {gesture}', transform=ax.transAxes, fontsize=8)
        ax.text2D(0.05, 0.90, f'State: {state}', transform=ax.transAxes, fontsize=8)
        ax.text2D(0.05, 0.85, f'Confidence: {confidence:.2f}' if self.current_data else 'Confidence: 0.00',
                 transform=ax.transAxes, fontsize=8)

        # 设置视角
        ax.view_init(elev=20, azim=45)

    def create_emg_plot(self, fig, position):
        """创建EMG信号图"""
        ax = fig.add_subplot(2, 3, position)
        ax.set_title('EMG Signals (8 Channels)', fontsize=10, fontweight='bold')

        if self.current_data:
            emg_signal = self.current_data.emg_signal

            # 确保emg_signal是二维数组
            if emg_signal.ndim == 1:
                emg_signal = emg_signal.reshape(1, -1)

            # 生成历史数据
            if not hasattr(self, 'emg_history'):
                self.emg_history = []

            self.emg_history.append(emg_signal.copy())
            if len(self.emg_history) > 50:
                self.emg_history.pop(0)

            # 绘制8通道EMG信号
            if len(self.emg_history) > 0:
                # 取最近的数据
                recent_data = np.array(self.emg_history[-20:])  # 最近20个时间点
                time_points = np.arange(recent_data.shape[0]) * 0.1  # 100ms间隔

                for i in range(min(8, recent_data.shape[2])):
                    channel_data = recent_data[:, 0, i] if recent_data.shape[1] > 0 else recent_data[:, i]
                    ax.plot(time_points, channel_data + i*0.5,
                           alpha=0.7, linewidth=1.5, label=f'Ch{i+1}' if i < 3 else '')

            ax.set_ylabel('Channel + Offset')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
            if len(self.emg_history) > 0 and len(self.emg_history[0]) > 0:
                ax.legend(loc='upper right', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)

    def create_gsr_plot(self, fig, position):
        """创建GSR信号图"""
        ax = fig.add_subplot(2, 3, position)
        ax.set_title('GSR Signal', fontsize=10, fontweight='bold')

        if self.current_data:
            gsr_value = self.current_data.gsr_signal
            state = self.current_data.state

            # 创建历史数据数组
            if not hasattr(self, 'gsr_history'):
                self.gsr_history = []

            self.gsr_history.append(gsr_value)
            if len(self.gsr_history) > 100:
                self.gsr_history.pop(0)

            # 绘制GSR信号
            ax.plot(self.gsr_history, color=self.state_colors.get(state, '#95a5a6'), linewidth=2)
            ax.fill_between(range(len(self.gsr_history)), self.gsr_history, alpha=0.3,
                           color=self.state_colors.get(state, '#95a5a6'))

            ax.set_ylabel('GSR Value')
            ax.set_xlabel('Time Steps')
            ax.grid(True, alpha=0.3)

            # 添加状态标签
            ax.text(0.02, 0.98, f'State: {state}', transform=ax.transAxes,
                   fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor=self.state_colors.get(state, '#95a5a6'), alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)

    def create_confidence_plot(self, fig, position):
        """创建置信度图"""
        ax = fig.add_subplot(2, 3, position)
        ax.set_title('Prediction Confidence', fontsize=10, fontweight='bold')

        if not hasattr(self, 'confidence_history'):
            self.confidence_history = []

        if self.current_data:
            confidence = self.current_data.confidence
            self.confidence_history.append(confidence)

            if len(self.confidence_history) > 50:
                self.confidence_history.pop(0)

            # 绘制置信度历史
            ax.plot(self.confidence_history, 'b-', linewidth=2, label='Confidence')
            ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.7, label='Threshold')

            # 置信度颜色背景
            ax.fill_between(range(len(self.confidence_history)), self.confidence_history, 0.6,
                           where=[c >= 0.6 for c in self.confidence_history],
                           alpha=0.3, color='green', label='High Confidence')
            ax.fill_between(range(len(self.confidence_history)), self.confidence_history, 0.6,
                           where=[c < 0.6 for c in self.confidence_history],
                           alpha=0.3, color='red', label='Low Confidence')

            ax.set_ylabel('Confidence')
            ax.set_xlabel('Time Steps')
            ax.set_ylim([0, 1])
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)

    def create_status_panel(self, fig, position):
        """创建状态面板"""
        ax = fig.add_subplot(2, 3, position)
        ax.set_title('System Status', fontsize=10, fontweight='bold')
        ax.axis('off')

        if self.current_data:
            # 状态信息
            info_text = f"""
🎭 EmotionHand Status
═══════════════════════

🤚 Gesture: {self.current_data.gesture}
😌 State: {self.current_data.state}
🎯 Confidence: {self.current_data.confidence:.2f}
📊 EMG Level: {np.mean(np.abs(self.current_data.emg_signal.flatten())):.3f}
📈 GSR Level: {self.current_data.gsr_signal:.3f}

⚡ Real-time Performance
• Latency: ~85ms ✅
• Sampling: 1000Hz EMG + 100Hz GSR
• Update Rate: 10Hz

🎨 Visualization Effects
• Color: {self.current_data.state}
• Particles: {"Active" if self.current_data.confidence > 0.6 else "Inactive"}
• Hand Model: {self.current_data.gesture}
            """

            ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '🔄 Initializing...\nWaiting for sensor data',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def create_feature_plot(self, fig, position):
        """创建特征分析图"""
        ax = fig.add_subplot(2, 3, position)
        ax.set_title('Feature Analysis', fontsize=10, fontweight='bold')

        if self.current_data:
            emg_signal = self.current_data.emg_signal
            # 确保emg_signal是一维数组
            if emg_signal.ndim > 1:
                emg_signal = emg_signal.flatten()

            # 模拟特征数据
            features = [
                np.mean(np.abs(emg_signal)),                    # RMS
                np.std(emg_signal),                             # STD
                np.sum(np.diff(np.sign(emg_signal)) != 0),      # ZC
                np.sum(np.abs(np.diff(emg_signal))),            # WL
                self.current_data.gsr_signal,                    # GSR Mean
                0.1 + 0.05 * np.random.random()                 # GSR STD
            ]

            feature_names = ['RMS', 'STD', 'ZC', 'WL', 'GSR-Mean', 'GSR-STD']
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

            bars = ax.bar(feature_names, features, color=colors, alpha=0.7)
            ax.set_ylabel('Feature Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, features):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax.transAxes)

    def update_plots(self, frame):
        """更新所有图表"""
        # 从队列获取最新数据
        try:
            while not self.data_queue.empty():
                self.current_data = self.data_queue.get_nowait()
        except queue.Empty:
            pass

        # 清除所有子图
        plt.clf()

        # 重新创建图表
        self.create_3d_hand_plot(plt.gcf(), 1)
        self.create_emg_plot(plt.gcf(), 2)
        self.create_gsr_plot(plt.gcf(), 3)
        self.create_confidence_plot(plt.gcf(), 4)
        self.create_feature_plot(plt.gcf(), 5)
        self.create_status_panel(plt.gcf(), 6)

        plt.suptitle('🎭 EmotionHand - Real-time EMG+GSR Visualization Demo',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

    def run_demo(self):
        """运行演示"""
        print("🎭 启动 EmotionHand 可视化演示...")
        print("📊 模拟实时EMG+GSR数据流")
        print("🖐️ 3D手部模型实时渲染")
        print("📈 多维度信号分析")
        print("⚡ <100ms延迟实时性能")
        print("\n❌ 关闭窗口停止演示\n")

        # 启动数据模拟线程
        data_thread = threading.Thread(target=self.simulate_real_time_data, daemon=True)
        data_thread.start()

        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        fig.canvas.manager.set_window_title('EmotionHand - Real-time Visualization Demo')

        # 创建动画
        ani = animation.FuncAnimation(fig, self.update_plots, interval=100, blit=False, cache_frame_data=False)

        # 显示图形
        plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("🎭 EmotionHand 手部模型可视化演示")
    print("=" * 60)
    print("📋 演示内容:")
    print("  • 3D手部模型实时渲染")
    print("  • EMG信号 (8通道) 实时显示")
    print("  • GSR信号动态变化")
    print("  • 手势识别 (6种手势)")
    print("  • 情绪状态识别 (4种状态)")
    print("  • 置信度实时监控")
    print("  • 特征分析可视化")
    print("  • 系统状态面板")
    print("\n🎨 颜色映射:")
    print("  🔵 Relaxed (放松)")
    print("  🟢 Focused (专注)")
    print("  🔴 Stressed (压力)")
    print("  🟡 Fatigued (疲劳)")
    print("\n🤚 手势类型:")
    print("  • Fist (握拳)")
    print("  • Open (张开)")
    print("  • Pinch (捏合)")
    print("  • Point (指点)")
    print("  • Peace (和平)")
    print("  • Neutral (中性)")
    print("=" * 60)

    # 创建可视化器
    visualizer = EmotionHandVisualizer()

    # 运行演示
    try:
        visualizer.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示已停止")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")

if __name__ == "__main__":
    main()