#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 3D手势可视化 - 修复版本
修复字体和坐标轴问题
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import time
import random
import json
from pathlib import Path

# 强制设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 确保中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS', 'Helvetica']

def load_config():
    """加载配置"""
    config_file = '3d_visualization_config.json'
    if Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            pass

    # 默认配置
    return {
        "palm_length": 0.6,
        "palm_width": 0.6,
        "finger_lengths": [0.4, 0.5, 0.45, 0.35],
        "gesture_bends": {
            "Fist": [80, 75, 70, 65],
            "Open": [10, 10, 10, 10],
            "Pinch": [15, 70, 75, 80],
            "Point": [15, 15, 15, 70],
            "Peace": [15, 15, 15, 15],
            "Neutral": [25, 25, 25, 25]
        },
        "state_colors": {
            "Relaxed": "#3498db",
            "Focused": "#2ecc71",
            "Stressed": "#e74c3c",
            "Fatigued": "#f39c12"
        },
        "update_interval": 50,
        "animation_fps": 15
    }

class Hand3D:
    """3D手势类"""

    def __init__(self, config):
        self.config = config

    def draw_hand(self, ax, gesture="Open", state="Relaxed", confidence=0.7):
        """绘制3D手势"""
        # 清除现有图形
        ax.clear()

        # 手势参数
        bend_angles = self.config["gesture_bends"].get(gesture, [25, 25, 25, 25])
        state_color = self.config["state_colors"].get(state, "#95a5a6")
        gesture_color = self.config.get("gesture_colors", {}).get(gesture, "#95a5a6")

        # 手掌参数
        palm_length = self.config["palm_length"]
        palm_width = self.config["palm_width"]
        finger_lengths = self.config["finger_lengths"]

        # 手掌顶点
        palm_vertices = np.array([
            [-palm_width/2, -palm_width/2, 0],
            [palm_width/2, palm_width/2, 0],
            [palm_width/2, palm_width/2, 0],
            [-palm_width/2, palm_width/2, 0]
        ])

        # 手掌连接
        palm_connections = [
            [0, 1], [1, 2], [2, 3], [3, 0]
        ]

        # 绘制手掌
        for connection in palm_connections:
            points = palm_vertices[connection]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            ax.plot(xs, ys, zs, color=state_color, linewidth=3, alpha=0.7)

        # 手掌底面
        palm_bottom = np.array([
            [-palm_width/2, -palm_width/2, -palm_length/3],
            [palm_width/2, palm_width/2, -palm_length/3],
            [palm_width/2, -palm_width/2, -palm_length/3],
            [-palm_width/2, -palm_width/2, -palm_length/3]
        ])

        # 手指绘制
        finger_base_positions = [
            [-palm_width/3, 0, 0],      # 小指
            [-palm_width/6, 0, 0],      # 无名指
            [0, 0, 0],               # 中指
            [palm_width/6, 0, 0],       # 食指
            [palm_width/3, 0, 0]        # 大指
        ]

        # 绘制每根手指
        for i in range(4):
            base_pos = finger_base_positions[i]
            finger_length = finger_lengths[i] if i < 4 else finger_lengths[3] * 0.7  # 大指稍短

            # 计算弯曲
            bend_angle_rad = np.radians(bend_angles[i])
            num_segments = 8

            # 手指关节点
            finger_points = [base_pos]
            for j in range(num_segments):
                t = j / (num_segments - 1)

                # 弯曲计算
                bend_x = np.cos(bend_angle_rad) * np.sin(t * np.pi/2) * 0.15
                bend_y = np.sin(bend_angle_rad) * np.sin(t * np.pi/2) * 0.15
                bend_z = np.cos(t * np.pi/2) * 0.2 * 0.5 if i == 4 else 0.3

                # 渐进位置
                progress = j / num_segments
                current_x = base_pos[0] + finger_length * progress * np.cos(t * np.pi/2) * 0.15
                current_y = base_pos[1] + finger_length * progress * np.sin(t * np.pi/2) * 0.15
                current_z = base_pos[2] + finger_length * progress * 0.3

                finger_points.append([current_x + bend_x, current_y + bend_y, current_z + bend_z])

            # 绘制手指线条
            finger_points = np.array(finger_points)
            ax.plot(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2],
                   color=gesture_color, linewidth=5, alpha=0.8)

            # 绘制关节
            ax.scatter(finger_points[:, 0], finger_points[:, 1], finger_points[:, 2],
                      color=state_color, s=30, alpha=0.9, edgecolors='black')

        # 设置坐标轴
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # 设置标题和标签
        ax.set_title(f'3D手势: {gesture}', fontsize=14, fontweight='bold', color=state_color)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)

        # 添加状态信息
        info_text = f'状态: {state}\\n置信度: {confidence:.2f}'
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    """主函数"""
    print("🎭 EmotionHand 3D手势可视化 - 修复版本")
    print("=" * 50)
    print("✅ 修复内容:")
    print("  • 中文字体显示问题已修复")
    print("  • 坐标轴范围优化")
    print("  • 更清晰的3D效果")
    print("=" * 50)

    # 加载配置
    config = load_config()
    hand_3d = Hand3D(config)

    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 手势列表
    gestures = list(config["gesture_bends"].keys())
    current_gesture_idx = 0

    def update(frame):
        nonlocal current_gesture_idx
        gesture = gestures[current_gesture_idx]
        state = random.choice(list(config["state_colors"].keys()))
        confidence = random.uniform(0.5, 1.0)

        hand_3d.draw_hand(ax, gesture, state, confidence)
        ax.view_init(elev=20, azim=45)

        # 更新手势
        current_gesture_idx = (current_gesture_idx + 1) % len(gestures)
        return ax,

    # 创建动画
    anim = plt.animation.FuncAnimation(
        fig, update, frames=len(gestures)*2,
        interval=1000//config["animation_fps"],  # milliseconds
        blit=False, repeat=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()