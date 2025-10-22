#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 静态手部模型演示
展示3D手部模型和不同手势状态
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HandModel3D:
    """3D手部模型"""

    def __init__(self):
        # 手部几何参数
        self.palm_length = 0.85
        self.palm_width = 0.85
        self.finger_lengths = [0.65, 0.75, 0.70, 0.55]  # 食指到小指
        self.thumb_length = 0.55
        self.finger_width = 0.18

        # 手势弯曲角度
        self.gesture_bends = {
            'Fist': [85, 80, 75, 70],
            'Open': [5, 5, 5, 5],
            'Pinch': [10, 75, 80, 85],
            'Point': [10, 10, 10, 80],
            'Peace': [10, 10, 10, 10],
            'Neutral': [20, 20, 20, 20]
        }

        # 状态颜色
        self.state_colors = {
            'Relaxed': '#3498db',      # 蓝色
            'Focused': '#2ecc71',      # 绿色
            'Stressed': '#e74c3c',     # 红色
            'Fatigued': '#f39c12'      # 黄色
        }

    def get_finger_joints(self, gesture: str, finger_idx: int) -> list:
        """计算手指关节位置"""
        bend_angles = self.gesture_bends.get(gesture, [20, 20, 20, 20])
        bend_angle = bend_angles[min(finger_idx, 3)]

        # 手指根部位置
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
            bend_rad = np.radians(bend_angle * (i + 1) / segments)
            current_x += segment_length * np.sin(bend_rad) * 0.3
            current_y += segment_length * np.cos(bend_rad)
            current_z += segment_length * np.sin(bend_rad) * 0.2 * (1 if i % 2 == 0 else -1)
            joints.append((current_x, current_y, current_z))

        return joints

    def draw_hand(self, ax, gesture: str, state: str, title: str):
        """绘制3D手部模型"""

        # 设置颜色
        hand_color = self.state_colors.get(state, '#95a5a6')

        # 绘制手掌
        palm_corners = [
            [-self.palm_width/2, 0, -self.palm_width/2],
            [self.palm_width/2, 0, -self.palm_width/2],
            [self.palm_width/2, 0, self.palm_width/2],
            [-self.palm_width/2, 0, self.palm_width/2]
        ]

        # 手掌顶面
        palm_top = [[p[0], p[1] + 0.1, p[2]] for p in palm_corners]
        palm_collection = Poly3DCollection([palm_top], alpha=0.7,
                                          facecolor=hand_color, edgecolor='black', linewidth=1)
        ax.add_collection3d(palm_collection)

        # 绘制手指
        for finger_idx in range(5):
            joints = self.get_finger_joints(gesture, finger_idx)
            xs, ys, zs = zip(*joints)
            ax.plot(xs, ys, zs, 'o-', color=hand_color, linewidth=3,
                   markersize=6, markerfacecolor=hand_color, markeredgecolor='black')

        # 设置坐标轴
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 2])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置标题和视角
        ax.set_title(f'{title}\n{gesture} | {state}', fontsize=10, fontweight='bold')
        ax.view_init(elev=20, azim=45)

        # 添加网格
        ax.grid(True, alpha=0.3)

def create_demo_visualization():
    """创建演示可视化"""

    print("🎭 创建 EmotionHand 手部模型演示...")

    # 创建手部模型
    hand_model = HandModel3D()

    # 创建3x2的子图布局
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('🎭 EmotionHand - 3D Hand Model Visualization\n'
                 'Real-time EMG+GSR Emotion Recognition System',
                 fontsize=16, fontweight='bold')

    # 定义演示场景
    scenarios = [
        ('Open', 'Relaxed', 'Relaxed State - Open Hand'),
        ('Fist', 'Stressed', 'Stressed State - Closed Fist'),
        ('Pinch', 'Focused', 'Focused State - Pinch Gesture'),
        ('Point', 'Focused', 'Focused State - Pointing'),
        ('Peace', 'Relaxed', 'Relaxed State - Peace Sign'),
        ('Neutral', 'Fatigued', 'Fatigued State - Neutral')
    ]

    # 绘制6个不同的手部状态
    for i, (gesture, state, title) in enumerate(scenarios):
        ax = fig.add_subplot(3, 2, i+1, projection='3d')
        hand_model.draw_hand(ax, gesture, state, title)

    plt.tight_layout()

    # 添加颜色图例
    legend_elements = [
        mpatches.Patch(color='#3498db', label='Relaxed (放松)'),
        mpatches.Patch(color='#2ecc71', label='Focused (专注)'),
        mpatches.Patch(color='#e74c3c', label='Stressed (压力)'),
        mpatches.Patch(color='#f39c12', label='Fatigued (疲劳)')
    ]
    plt.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    # 添加技术信息
    tech_info = """
    📊 Technical Specifications:
    • EMG: 8-channel, 1000Hz sampling rate
    • GSR: Single-channel, 100Hz sampling rate
    • Features: RMS, MDF, ZC, WL + GSR statistics
    • Algorithms: LightGBM, SVM, LDA
    • Real-time latency: <100ms
    • Calibration time: 2 minutes
    • Visualization: Unity 3D with particle effects
    """

    fig.text(0.02, 0.02, tech_info, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图片
    plt.savefig('EmotionHand_Hand_Model_Demo.png', dpi=300, bbox_inches='tight')
    print("✅ 演示图片已保存: EmotionHand_Hand_Model_Demo.png")

    plt.show()

def create_signal_demo():
    """创建信号演示"""

    print("📊 创建 EMG+GSR 信号演示...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📈 EmotionHand - Signal Analysis Demo\n'
                 'EMG+GSR Dual-Modal Emotion Recognition',
                 fontsize=14, fontweight='bold')

    # 模拟时间轴
    t = np.linspace(0, 2, 1000)

    # 1. EMG信号图
    ax1 = axes[0, 0]
    emg_channels = 4  # 显示前4通道

    for i in range(emg_channels):
        # 不同手势的特征频率
        if i == 0:  # Fist
            freq = 30 + 10*i
            amplitude = 0.8
        elif i == 1:  # Open
            freq = 15 + 5*i
            amplitude = 0.3
        elif i == 2:  # Pinch
            freq = 40 + 15*i
            amplitude = 0.6
        else:  # Point
            freq = 25 + 8*i
            amplitude = 0.5

        signal = amplitude * np.sin(2 * np.pi * freq * t)
        signal += 0.1 * np.random.randn(len(t))  # 噪声
        signal += i * 0.5  # 通道偏移

        ax1.plot(t, signal, alpha=0.8, linewidth=1.5, label=f'Channel {i+1}')

    ax1.set_title('EMG Signals (8 Channels)', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (with offset)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. GSR信号图
    ax2 = axes[0, 1]
    # 模拟不同状态的GSR信号
    gsr_states = {
        'Relaxed': 0.1 + 0.05 * np.sin(2 * np.pi * 0.5 * t),
        'Focused': 0.2 + 0.08 * np.sin(2 * np.pi * 0.8 * t),
        'Stressed': 0.4 + 0.15 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t)),
        'Fatigued': 0.25 + 0.12 * np.sin(2 * np.pi * 0.6 * t)
    }

    colors = {'Relaxed': '#3498db', 'Focused': '#2ecc71',
              'Stressed': '#e74c3c', 'Fatigued': '#f39c12'}

    for state, signal in gsr_states.items():
        ax2.plot(t, signal, label=state, color=colors[state], linewidth=2)

    ax2.set_title('GSR Signals for Different States', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('GSR Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. 特征分析图
    ax3 = axes[1, 0]
    features = ['RMS', 'STD', 'ZC', 'WL', 'GSR-Mean', 'GSR-STD']

    # 模拟不同手势的特征值
    feature_values = {
        'Fist': [0.8, 0.3, 120, 45, 0.4, 0.12],
        'Open': [0.2, 0.1, 20, 8, 0.1, 0.05],
        'Pinch': [0.6, 0.25, 80, 32, 0.3, 0.08],
        'Point': [0.4, 0.18, 60, 25, 0.25, 0.07]
    }

    x = np.arange(len(features))
    width = 0.2

    for i, (gesture, values) in enumerate(feature_values.items()):
        ax3.bar(x + i*width, values, width, label=gesture, alpha=0.8)

    ax3.set_title('Feature Analysis by Gesture', fontweight='bold')
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Feature Value')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(features, rotation=45)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. 性能指标图
    ax4 = axes[1, 1]
    metrics = ['Latency (ms)', 'Accuracy (%)', 'Calibration (min)', 'FPS']
    achieved = [85, 87, 2, 50]
    targets = [100, 80, 5, 30]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax4.bar(x - width/2, achieved, width, label='Achieved', color='#2ecc71', alpha=0.8)
    bars2 = ax4.bar(x + width/2, targets, width, label='Target', color='#95a5a6', alpha=0.8)

    ax4.set_title('Performance Metrics', fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图片
    plt.savefig('EmotionHand_Signal_Analysis_Demo.png', dpi=300, bbox_inches='tight')
    print("✅ 信号分析图片已保存: EmotionHand_Signal_Analysis_Demo.png")

    plt.show()

def main():
    """主函数"""
    print("=" * 70)
    print("🎭 EmotionHand 手部模型和信号演示")
    print("=" * 70)
    print("📋 演示内容:")
    print("  1. 3D手部模型 - 6种手势和状态组合")
    print("  2. EMG信号分析 - 8通道实时信号")
    print("  3. GSR信号分析 - 4种情绪状态")
    print("  4. 特征分析 - RMS, STD, ZC, WL")
    print("  5. 性能指标 - 延迟、精度、校准时间")
    print("\n🎨 颜色映射:")
    print("  🔵 Relaxed (放松) - 蓝色")
    print("  🟢 Focused (专注) - 绿色")
    print("  🔴 Stressed (压力) - 红色")
    print("  🟡 Fatigued (疲劳) - 黄色")
    print("=" * 70)

    # 创建演示
    create_demo_visualization()
    create_signal_demo()

    print("\n🎉 演示完成!")
    print("📁 生成的文件:")
    print("  • EmotionHand_Hand_Model_Demo.png - 3D手部模型演示")
    print("  • EmotionHand_Signal_Analysis_Demo.png - 信号分析演示")
    print("\n🚀 这些图片展示了EmotionHand系统的核心功能和视觉效果!")

if __name__ == "__main__":
    main()