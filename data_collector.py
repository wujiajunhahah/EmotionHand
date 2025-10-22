#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 数据采集器
用于收集真实的EMG+GSR数据来训练模型

使用方法:
python data_collector.py --output training_data.csv --duration 300
"""

import numpy as np
import pandas as pd
import time
import argparse
import threading
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataCollector:
    """真实数据采集器"""

    def __init__(self, config_file: str = 'emotionhand_config.json'):
        self.config = self._load_config(config_file)
        self.running = False
        self.data_buffer = []
        self.start_time = None

    def _load_config(self, config_file: str) -> Dict:
        """加载配置"""
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}")

        # 默认配置
        return {
            'emg_sample_rate': 1000,
            'gsr_sample_rate': 100,
            'window_size': 256
        }

    def simulate_hardware_input(self) -> Dict:
        """模拟硬件输入（用于测试）"""
        if not hasattr(self, 'simulation_time'):
            self.simulation_time = 0

        self.simulation_time += 0.1

        # 模拟EMG信号（8通道）
        emg_signal = np.random.randn(8) * 0.5
        emg_signal[0] += 0.3 * np.sin(self.simulation_time * 2)
        emg_signal[1] += 0.2 * np.sin(self.simulation_time * 3)

        # 模拟GSR信号
        gsr_value = 0.2 + 0.1 * np.sin(self.simulation_time * 0.5) + 0.05 * np.random.randn()

        return {
            'emg': emg_signal.tolist(),
            'gsr': float(gsr_value),
            'timestamp': time.time()
        }

    def extract_emg_features(self, emg_signal: List[float]) -> List[float]:
        """提取EMG特征"""
        signal = np.array(emg_signal)

        # RMS - 均方根
        rms = float(np.sqrt(np.mean(signal ** 2)))

        # STD - 标准差
        std = float(np.std(signal))

        # ZC - 过零率
        zc = int(np.sum(np.diff(np.sign(signal)) != 0))

        # WL - 波长长度
        wl = float(np.sum(np.abs(np.diff(signal))))

        return [rms, std, zc, wl]

    def collect_data_session(self, duration: int = 300, output_file: str = 'collected_data.csv'):
        """采集数据会话"""
        logger.info(f"开始数据采集，持续时间: {duration}秒")
        logger.info(f"输出文件: {output_file}")

        self.running = True
        self.start_time = time.time()
        session_data = []

        def data_collection_thread():
            while self.running and (time.time() - self.start_time) < duration:
                # 获取传感器数据
                sensor_data = self.simulate_hardware_input()

                # 提取特征
                emg_features = self.extract_emg_features(sensor_data['emg'])

                # 用户手动标注（真实场景中）
                if len(session_data) % 50 == 0:  # 每50个样本提示一次
                    print(f"\n📋 数据进度: {len(session_data)} 样本")
                    print("请输入当前状态 (1=Relaxed, 2=Focused, 3=Stressed, 4=Fatigued):")
                    try:
                        state_input = input("状态编号 (回车跳过): ").strip()
                        if state_input:
                            state_map = {'1': 'Relaxed', '2': 'Focused', '3': 'Stressed', '4': 'Fatigued'}
                            if state_input in state_map:
                                current_state = state_map[state_input]
                            else:
                                print("无效输入，跳过此样本")
                                current_state = 'Unknown'
                        else:
                            current_state = 'Unknown'
                    except KeyboardInterrupt:
                        print("\n用户中断")
                        self.running = False
                        return
                else:
                    current_state = 'Unknown'

                # 手势（基于EMG特征简单判断）
                rms_value = emg_features[0]
                if rms_value > 0.6:
                    current_gesture = 'Fist'
                elif rms_value > 0.3:
                    current_gesture = 'Pinch'
                else:
                    current_gesture = 'Open'

                # 保存数据
                data_point = {
                    'timestamp': sensor_data['timestamp'],
                    'emg_raw': str(sensor_data['emg']),  # 保存为字符串
                    'gsr_raw': sensor_data['gsr'],
                    'rms': emg_features[0],
                    'std': emg_features[1],
                    'zc': emg_features[2],
                    'wl': emg_features[3],
                    'gesture': current_gesture,
                    'state': current_state,
                    'confidence': 0.8 if current_state != 'Unknown' else 0.3
                }

                session_data.append(data_point)
                time.sleep(0.1)  # 10Hz采集率

        # 启动数据采集线程
        collection_thread = threading.Thread(target=data_collection_thread, daemon=True)
        collection_thread.start()

        try:
            # 等待采集完成
            while self.running and (time.time() - self.start_time) < duration:
                time.sleep(1)
                remaining = int(duration - (time.time() - self.start_time))
                if remaining % 10 == 0:
                    logger.info(f"剩余时间: {remaining}秒，已采集: {len(session_data)}样本")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断采集")
        finally:
            self.running = False

        # 保存数据
        self._save_collected_data(session_data, output_file)

    def _save_collected_data(self, data: List[Dict], output_file: str):
        """保存采集的数据"""
        if not data:
            logger.warning("没有采集到数据")
            return

        # 过滤有效数据
        valid_data = [d for d in data if d['state'] != 'Unknown']

        logger.info(f"总共采集: {len(data)} 样本")
        logger.info(f"有效样本: {len(valid_data)} 样本")
        logger.info(f"无效样本: {len(data) - len(valid_data)} 样本")

        if valid_data:
            df = pd.DataFrame(valid_data)
            df.to_csv(output_file, index=False)
            logger.info(f"✅ 数据已保存到: {output_file}")

            # 显示统计信息
            print(f"\n📊 数据统计:")
            print(f"  总样本数: {len(data)}")
            print(f"  有效样本数: {len(valid_data)}")
            print(f"  状态分布:")
            state_counts = df['state'].value_counts()
            for state, count in state_counts.items():
                print(f"    {state}: {count} 样本")
            print(f"  手势分布:")
            gesture_counts = df['gesture'].value_counts()
            for gesture, count in gesture_counts.items():
                print(f"    {gesture}: {count} 样本")
        else:
            logger.warning("没有有效数据可保存")

def generate_predefined_dataset(output_file: str = 'predefined_emotion_data.csv'):
    """生成预定义的数据集（用于快速测试）"""
    logger.info("生成预定义训练数据集...")

    # 定义特征范围
    np.random.seed(42)
    states = ['Relaxed', 'Focused', 'Stressed', 'Fatigued']
    gestures = ['Open', 'Pinch', 'Fist']

    data = []

    # 为每个状态-手势组合生成样本
    for state in states:
        for gesture in gestures:
            # 每个组合生成30个样本
            for _ in range(30):
                # 基于状态和手势生成特征
                if state == 'Relaxed':
                    rms = np.random.normal(0.2, 0.1)
                    std = np.random.normal(0.1, 0.05)
                    zc = np.random.randint(5, 20)
                    wl = np.random.normal(10, 5)
                    gsr = np.random.normal(0.15, 0.05)
                elif state == 'Focused':
                    rms = np.random.normal(0.5, 0.15)
                    std = np.random.normal(0.3, 0.1)
                    zc = np.random.randint(20, 50)
                    wl = np.random.normal(30, 10)
                    gsr = np.random.normal(0.25, 0.08)
                elif state == 'Stressed':
                    rms = np.random.normal(0.8, 0.2)
                    std = np.random.normal(0.5, 0.15)
                    zc = np.random.randint(50, 100)
                    wl = np.random.normal(60, 15)
                    gsr = np.random.normal(0.4, 0.12)
                else:  # Fatigued
                    rms = np.random.normal(0.3, 0.12)
                    std = np.random.normal(0.2, 0.08)
                    zc = np.random.randint(10, 30)
                    wl = np.random.normal(20, 8)
                    gsr = np.random.normal(0.3, 0.1)

                # 添加噪声增加真实性
                rms += np.random.normal(0, 0.02)
                std += np.random.normal(0, 0.01)
                gsr += np.random.normal(0, 0.01)

                data_point = {
                    'timestamp': time.time(),
                    'rms': max(0, rms),
                    'std': max(0, std),
                    'zc': max(0, zc),
                    'wl': max(0, wl),
                    'gsr': max(0, gsr),
                    'gesture': gesture,
                    'state': state,
                    'confidence': np.random.uniform(0.6, 0.95)
                }
                data.append(data_point)

    # 保存数据
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    logger.info(f"✅ 预定义数据集已保存到: {output_file}")
    logger.info(f"总共生成: {len(data)} 个样本")

    # 显示数据分布
    print(f"\n📊 生成数据分布:")
    print(f"  状态分布:")
    state_counts = df['state'].value_counts()
    for state, count in state_counts.items():
        print(f"    {state}: {count} 样本")
    print(f"  手势分布:")
    gesture_counts = df['gesture'].value_counts()
    for gesture, count in gesture_counts.items():
        print(f"    {gesture}: {count} 样本")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EmotionHand 数据采集器')
    parser.add_argument('--output', type=str, default='collected_data.csv',
                      help='输出CSV文件路径')
    parser.add_argument('--duration', type=int, default=300,
                      help='采集持续时间（秒）')
    parser.add_argument('--generate-predefined', action='store_true',
                      help='生成预定义数据集')
    parser.add_argument('--predefined-file', type=str, default='predefined_emotion_data.csv',
                      help='预定义数据集文件名')

    args = parser.parse_args()

    print("🎭 EmotionHand 数据采集器")
    print("=" * 40)

    if args.generate_predefined:
        # 生成预定义数据集
        generate_predefined_dataset(args.predefined_file)
    else:
        # 真实数据采集
        collector = RealDataCollector()

        print("📋 数据采集说明:")
        print("  1. 采集过程中会提示输入情绪状态")
        print("  2. 状态编号: 1=Relaxed, 2=Focused, 3=Stressed, 4=Fatigued")
        print("  3. 每个状态建议采集至少50个样本")
        print("  4. 按 Ctrl+C 可提前结束采集")
        print("  5. 无传感器时使用模拟数据")
        print("=" * 40)

        try:
            collector.collect_data_session(args.duration, args.output)
        except Exception as e:
            logger.error(f"数据采集失败: {e}")

if __name__ == "__main__":
    main()