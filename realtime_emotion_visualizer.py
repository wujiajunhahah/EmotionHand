#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 实时情绪可视化系统
Real-time Emotion Visualization with Professional Signal Processing

集成专业级信号处理的实时可视化系统：
• 企业级信号处理引擎
• 个体化校准系统
• 低延迟情绪状态检测
• 3D手势可视化 + 质量监测面板

Author: EmotionHand Team
Version: 2.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import queue

# 导入自定义模块
from signal_processing_engine import RealTimeSignalProcessor, SignalQuality
from calibration_system import CalibrationSystem
from emotion_state_detector import EnsembleDetector, EmotionState

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

logger = logging.getLogger(__name__)


class EmotionVisualizationPanel:
    """情绪状态可视化面板"""

    def __init__(self, fig, ax_position):
        self.fig = fig
        self.ax = fig.add_axes(ax_position)
        self.ax.set_title('情绪状态监测', fontsize=14, fontweight='bold', pad=20)

        # 情绪状态配置
        self.emotion_config = {
            'Relaxed': {'color': '#3498db', 'y': 0.8},
            'Focused': {'color': '#2ecc71', 'y': 0.6},
            'Stressed': {'color': '#e74c3c', 'y': 0.4},
            'Fatigued': {'color': '#f39c12', 'y': 0.2}
        }

        # 历史数据
        self.state_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)

        self._setup_axes()

    def _setup_axes(self):
        """设置坐标轴"""
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('时间 (样本)', fontsize=10)
        self.ax.set_ylabel('置信度', fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # Y轴标签
        self.ax.set_yticks([config['y'] for config in self.emotion_config.values()])
        self.ax.set_yticklabels(list(self.emotion_config.keys()))

    def update(self, prediction, quality_status):
        """更新情绪状态面板"""
        # 添加到历史记录
        self.state_history.append(prediction.state.value)
        self.confidence_history.append(prediction.confidence)

        # 清除并重绘
        self.ax.clear()
        self._setup_axes()

        if len(self.state_history) > 1:
            # 绘制状态时间线
            x_data = list(range(len(self.state_history)))
            y_data = [self.emotion_config[state]['y'] for state in self.state_history]

            # 绘制连线
            for i in range(len(x_data) - 1):
                self.ax.plot(x_data[i:i+2], y_data[i:i+2],
                           color=self.emotion_config[self.state_history[i]]['color'],
                           alpha=0.7, linewidth=3)

            # 绘制数据点（大小基于置信度）
            for i, (x, y, conf, state) in enumerate(zip(x_data, y_data, self.confidence_history, self.state_history)):
                self.ax.scatter(x, y, s=50 + conf * 100,
                              color=self.emotion_config[state]['color'],
                              alpha=0.8, edgecolors='white', linewidth=1)

            # 标记当前状态
            current_color = self.emotion_config[prediction.state.value]['color']
            self.ax.scatter(x_data[-1], y_data[-1], s=200,
                          color=current_color, alpha=1.0,
                          edgecolors='white', linewidth=3, marker='o')

        # 显示当前状态信息
        info_text = (f"当前状态: {prediction.state.value}\n"
                    f"置信度: {prediction.confidence:.2f}\n"
                    f"推理: {prediction.reasoning}")

        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

        # 质量状态指示器
        quality_color = {'excellent': '#2ecc71', 'good': '#f39c12',
                        'poor': '#e67e22', 'bad': '#e74c3c'}.get(quality_status['status'], '#95a5a6')

        quality_rect = patches.Rectangle((0.7, 0.85), 0.25, 0.12,
                                       transform=self.ax.transAxes,
                                       facecolor=quality_color, alpha=0.8)
        self.ax.add_patch(quality_rect)

        self.ax.text(0.825, 0.91, f"质量: {quality_status['status']}",
                    transform=self.ax.transAxes, fontsize=9,
                    horizontalalignment='center', verticalalignment='center',
                    color='white', fontweight='bold')


class SignalQualityPanel:
    """信号质量监测面板"""

    def __init__(self, fig, ax_position):
        self.fig = fig
        self.ax = fig.add_axes(ax_position)
        self.ax.set_title('信号质量监测', fontsize=14, fontweight='bold', pad=20)

        # 质量历史
        self.emg_quality_history = deque(maxlen=100)
        self.gsr_quality_history = deque(maxlen=100)

        self._setup_axes()

    def _setup_axes(self):
        """设置坐标轴"""
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('时间 (样本)', fontsize=10)
        self.ax.set_ylabel('质量评分', fontsize=10)
        self.ax.grid(True, alpha=0.3)

        # 添加质量等级线
        self.ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='优秀')
        self.ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='良好')
        self.ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='警告')

    def update(self, emg_quality: SignalQuality, gsr_quality: SignalQuality):
        """更新质量监测面板"""
        # 添加到历史记录
        self.emg_quality_history.append(emg_quality.quality_score)
        self.gsr_quality_history.append(gsr_quality.quality_score)

        # 清除并重绘
        self.ax.clear()
        self._setup_axes()

        if len(self.emg_quality_history) > 1:
            x_data = list(range(len(self.emg_quality_history)))

            # 绘制质量曲线
            self.ax.plot(x_data, list(self.emg_quality_history),
                        color='#3498db', linewidth=2, label='EMG质量')
            self.ax.plot(x_data, list(self.gsr_quality_history),
                        color='#e74c3c', linewidth=2, label='GSR质量')

            # 当前质量点
            self.ax.scatter(x_data[-1], list(self.emg_quality_history)[-1],
                          color='#3498db', s=100, zorder=5)
            self.ax.scatter(x_data[-1], list(self.gsr_quality_history)[-1],
                          color='#e74c3c', s=100, zorder=5)

        # 图例
        self.ax.legend(loc='lower right', fontsize=9)

        # 质量统计信息
        if self.emg_quality_history:
            avg_emg = np.mean(list(self.emg_quality_history))
            avg_gsr = np.mean(list(self.gsr_quality_history))

            stats_text = (f"EMG: {avg_emg:.2f}\n"
                         f"GSR: {avg_gsr:.2f}\n"
                         f"SNR: {emg_quality.snr:.1f}dB")

            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))


class RealtimeEmotionVisualizer:
    """实时情绪可视化主系统"""

    def __init__(self, config_path: str = 'signal_processing_config.json'):
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化组件
        self.signal_processor = RealTimeSignalProcessor(config_path)
        self.calibration_system = CalibrationSystem(self.config)
        self.emotion_detector = EnsembleDetector(self.config)

        # 启动信号处理器
        self.signal_processor.start()

        # 创建可视化界面
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('EmotionHand 实时情绪监测系统', fontsize=16, fontweight='bold')

        # 面板布局
        self.emotion_panel = EmotionVisualizationPanel(self.fig, [0.05, 0.55, 0.4, 0.35])
        self.quality_panel = SignalQualityPanel(self.fig, [0.05, 0.10, 0.4, 0.35])

        # 3D手势可视化（使用现有的3D系统）
        self._setup_3d_visualization()

        # 性能监控
        self.fps_history = deque(maxlen=30)
        self.last_update_time = time.time()

        # 数据生成
        self.data_generator = self._create_data_generator()

        # 运行状态
        self.running = False

        logger.info("实时情绪可视化系统初始化完成")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return {}

    def _setup_3d_visualization(self):
        """设置3D手势可视化"""
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')
        self.ax_3d.set_title('3D手势可视化', fontsize=14, fontweight='bold')

        # 手势参数配置
        self.gesture_params = {
            'Fist': {'fingers': [85, 80, 75, 70], 'intensity': 0.9, 'color': '#8e44ad'},
            'Open': {'fingers': [5, 5, 5, 5], 'intensity': 0.2, 'color': '#95a5a6'},
            'Pinch': {'fingers': [10, 75, 80, 85], 'intensity': 0.7, 'color': '#e67e22'},
            'Point': {'fingers': [10, 10, 10, 80], 'intensity': 0.6, 'color': '#16a085'},
            'Neutral': {'fingers': [20, 20, 20, 20], 'intensity': 0.4, 'color': '#34495e'}
        }

    def _create_data_generator(self):
        """创建模拟数据生成器"""
        class DataGenerator:
            def __init__(self, config):
                self.config = config
                self.time = 0

            def generate_sample(self, emotion_state: str = "Neutral"):
                """生成模拟数据样本"""
                self.time += 0.1

                # 基于情绪状态生成EMG特征
                emotion_emg_params = {
                    'Relaxed': {'rms': 0.15, 'std': 0.08, 'mdf': 80},
                    'Focused': {'rms': 0.45, 'std': 0.25, 'mdf': 120},
                    'Stressed': {'rms': 0.75, 'std': 0.45, 'mdf': 150},
                    'Fatigued': {'rms': 0.25, 'std': 0.15, 'mdf': 60}
                }

                params = emotion_emg_params.get(emotion_state, emotion_emg_params['Neutral'])

                # 生成EMG数据
                emg_data = []
                for ch in range(8):
                    base_signal = params['rms'] * np.sin(self.time * (ch + 1) * 0.5)
                    noise = np.random.randn() * params['std']
                    emg_data.append(base_signal + noise)

                # 生成GSR数据
                gsr_base = 0.2 if emotion_state == 'Relaxed' else (0.35 if emotion_state in ['Focused', 'Stressed'] else 0.25)
                gsr_data = gsr_base + 0.05 * np.sin(self.time * 0.2) + 0.02 * np.random.randn()

                return emg_data, max(0, gsr_data)

        return DataGenerator(self.config)

    def draw_3d_hand(self, gesture: str, state: str, confidence: float):
        """绘制3D手势"""
        self.ax_3d.clear()

        # 设置3D视图
        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-1, 1])
        self.ax_3d.set_zlim([0, 2])
        self.ax_3d.set_xlabel('X', fontsize=10)
        self.ax_3d.set_ylabel('Y', fontsize=10)
        self.ax_3d.set_zlabel('Z', fontsize=10)

        # 手势参数
        gesture_config = self.gesture_params.get(gesture, self.gesture_params['Neutral'])
        emotion_config = self.emotion_detector.emotion_config

        # 手掌位置
        palm_vertices = self._generate_palm()

        # 手指
        finger_positions = []
        finger_params = gesture_config['fingers']

        for i, bend_angle in enumerate(finger_params):
            # 计算手指弯曲
            base_pos = [0.3 + i * 0.15, 0, 1]
            tip_z = 1.5 - (bend_angle / 90) * 0.5
            finger_positions.append([base_pos[0], base_pos[1], tip_z])

        # 颜色设置
        emotion_color = emotion_config.get(state, {'color': '#95a5a6'})['color']
        hand_color = gesture_config['color']

        # 动态透明度
        alpha = 0.3 + 0.7 * confidence

        # 绘制手掌
        palm_color = (*self._hex_to_rgb(hand_color), alpha * 0.6)
        for i in range(len(palm_vertices)):
            next_i = (i + 1) % len(palm_vertices)
            xs = [palm_vertices[i][0], palm_vertices[next_i][0]]
            ys = [palm_vertices[i][1], palm_vertices[next_i][1]]
            zs = [palm_vertices[i][2], palm_vertices[next_i][2]]
            self.ax_3d.plot(xs, ys, zs, color=emotion_color, linewidth=3, alpha=alpha)

        # 绘制手指
        for i, finger_pos in enumerate(finger_positions):
            # 手指基线
            self.ax_3d.plot([0.3 + i * 0.15, finger_pos[0]],
                          [0, finger_pos[1]],
                          [1, finger_pos[2]],
                          color=emotion_color, linewidth=8 - i, alpha=alpha)

            # 手指关节
            for j in range(1, 4):
                joint_z = 1 + (finger_pos[2] - 1) * (j / 3)
                self.ax_3d.scatter([0.3 + i * 0.15], [0], [joint_z],
                                 color=emotion_color, s=100, alpha=alpha * 0.8)

        # 添加状态标签
        self.ax_3d.text2D(0.05, 0.95, f'状态: {state}', transform=self.ax_3d.transAxes,
                         fontsize=12, fontweight='bold', color=emotion_color)
        self.ax_3d.text2D(0.05, 0.90, f'手势: {gesture}', transform=self.ax_3d.transAxes,
                         fontsize=10, color=emotion_color)

    def _generate_palm(self):
        """生成手掌顶点"""
        size = 0.4
        vertices = [
            [-size, -size, 1],
            [size, -size, 1],
            [size, size, 1],
            [-size, size, 1]
        ]
        return vertices

    def _hex_to_rgb(self, hex_color):
        """十六进制颜色转RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    def update_frame(self, frame):
        """更新帧数据"""
        try:
            # 生成模拟数据
            # 模拟情绪状态变化
            emotion_states = ['Relaxed', 'Focused', 'Stressed', 'Fatigued']
            current_emotion = emotion_states[frame % len(emotion_states)]

            emg_data, gsr_data = self.data_generator.generate_sample(current_emotion)

            # 添加到信号处理器
            self.signal_processor.add_data(emg_data, gsr_data)

            # 处理信号
            result = self.signal_processor.process_window()
            if result:
                # 情绪状态检测
                prediction = self.emotion_detector.predict_state(
                    result['normalized_features'],
                    result['emg_features'],
                    result['gsr_features']
                )

                # 更新各面板
                quality_status = self.signal_processor.get_quality_status()
                self.emotion_panel.update(prediction, quality_status)
                self.quality_panel.update(
                    SignalQuality(**result['quality']['emg']),
                    SignalQuality(**result['quality']['gsr'])
                )

                # 更新3D手势
                # 基于RMS值选择手势
                rms_value = result['normalized_features'].get('rms', 0.5)
                if rms_value > 0.6:
                    gesture = 'Fist'
                elif rms_value > 0.3:
                    gesture = 'Pinch'
                else:
                    gesture = 'Open'

                self.draw_3d_hand(gesture, prediction.state.value, prediction.confidence)

            # 性能监控
            current_time = time.time()
            frame_time = current_time - self.last_update_time
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            self.last_update_time = current_time

            # 显示FPS
            if len(self.fps_history) > 0:
                avg_fps = np.mean(list(self.fps_history))
                self.fig.suptitle(f'EmotionHand 实时情绪监测系统 - FPS: {avg_fps:.1f}',
                                fontsize=16, fontweight='bold')

        except Exception as e:
            logger.error(f"帧更新失败: {e}")

    def start_visualization(self):
        """启动可视化"""
        self.running = True
        self.animation = FuncAnimation(
            self.fig, self.update_frame,
            interval=int(1000 / self.config['realtime']['target_fps']),
            blit=False
        )
        plt.show()

    def stop_visualization(self):
        """停止可视化"""
        self.running = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()

    def show_performance_stats(self):
        """显示性能统计"""
        stats = self.signal_processor.get_performance_stats()
        print(f"\n🚀 性能统计:")
        print(f"  平均处理时间: {stats['avg_time']*1000:.1f}ms")
        print(f"  最大处理时间: {stats['max_time']*1000:.1f}ms")
        print(f"  处理FPS: {stats['fps']:.1f}")
        print(f"  延迟: {stats.get('latency_ms', stats['avg_time']*1000):.1f}ms")


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)

    print("🎭 EmotionHand 实时情绪可视化系统")
    print("=" * 50)

    try:
        # 创建可视化系统
        visualizer = RealtimeEmotionVisualizer()

        print("✅ 系统初始化完成")
        print("📊 正在启动实时监测...")
        print("💡 提示: 关闭窗口以退出系统")

        # 显示性能统计
        visualizer.show_performance_stats()

        # 启动可视化
        visualizer.start_visualization()

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        logger.exception("系统运行异常")
    finally:
        # 清理资源
        if 'visualizer' in locals():
            visualizer.stop_visualization()
            visualizer.signal_processor.stop()
        print("🔚 系统已关闭")


if __name__ == "__main__":
    main()