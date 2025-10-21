#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集脚本
Muscle Sensor v3 (EMG) + GSR传感器数据采集
支持手势和状态标注
"""

import os
import time
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
import json
import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    """EMG + GSR 数据采集器"""

    def __init__(self, config: Dict = None):
        # 默认配置
        self.config = {
            # 串口配置
            'emg_port': None,  # 自动检测
            'gsr_port': None,  # 自动检测
            'emg_baudrate': 115200,
            'gsr_baudrate': 9600,

            # 采样参数
            'emg_sample_rate': 1000,  # Hz
            'gsr_sample_rate': 100,   # Hz
            'collection_duration': 30, # 秒

            # 数据存储
            'output_dir': './data/private/raw',
            'session_name': f"session_{int(time.time())}",

            # 手势标签
            'gesture_labels': [
                'Fist',      # 握拳
                'Open',      # 张开
                'Pinch',     # 捏合
                'Point',     # 点按
                'Twist',     # 旋拧
                'Wave'       # 挥手
            ],

            # 状态标签
            'state_labels': [
                'Relaxed',   # 放松
                'Focused',   # 专注
                'Stressed',  # 压力
                'Fatigued'   # 疲劳
            ]
        }

        if config:
            self.config.update(config)

        # 数据存储
        self.emg_data = deque(maxlen=self.config['emg_sample_rate'] * 300)  # 5分钟缓冲
        self.gsr_data = deque(maxlen=self.config['gsr_sample_rate'] * 300)   # 5分钟缓冲
        self.timestamps = deque(maxlen=self.config['gsr_sample_rate'] * 300)

        # 当前采集状态
        self.current_label = None
        self.collecting = False
        self.collection_start_time = None

        # 串口连接
        self.emg_serial = None
        self.gsr_serial = None

        # 线程控制
        self.running = False
        self.threads = []

        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # 初始化连接
        self.init_connections()

        logger.info("数据采集器初始化完成")

    def init_connections(self):
        """初始化串口连接"""
        # 检测可用串口
        ports = serial.tools.list_ports.comports()
        logger.info(f"发现 {len(ports)} 个串口设备")

        for port in ports:
            logger.info(f"  {port.device}: {port.description}")

        # 自动检测EMG设备
        if not self.config['emg_port']:
            emg_ports = [p for p in ports if 'usbmodem' in p.device.lower() or 'acm' in p.device.lower()]
            if emg_ports:
                self.config['emg_port'] = emg_ports[0].device
                logger.info(f"自动检测EMG端口: {self.config['emg_port']}")

        # 自动检测GSR设备
        if not self.config['gsr_port']:
            gsr_ports = [p for p in ports if 'usbserial' in p.device.lower()]
            if gsr_ports:
                self.config['gsr_port'] = gsr_ports[0].device
                logger.info(f"自动检测GSR端口: {self.config['gsr_port']}")

        # 连接EMG设备
        if self.config['emg_port']:
            try:
                self.emg_serial = serial.Serial(
                    self.config['emg_port'],
                    self.config['emg_baudrate'],
                    timeout=1
                )
                logger.info(f"EMG设备连接成功: {self.config['emg_port']}")
            except Exception as e:
                logger.error(f"EMG设备连接失败: {e}")

        # 连接GSR设备
        if self.config['gsr_port']:
            try:
                self.gsr_serial = serial.Serial(
                    self.config['gsr_port'],
                    self.config['gsr_baudrate'],
                    timeout=1
                )
                logger.info(f"GSR设备连接成功: {self.config['gsr_port']}")
            except Exception as e:
                logger.error(f"GSR设备连接失败: {e}")

    def read_emg_data(self) -> Optional[np.ndarray]:
        """读取EMG数据"""
        if self.emg_serial and self.emg_serial.is_open:
            try:
                # 读取一行数据
                line = self.emg_serial.readline().decode('utf-8').strip()
                if line:
                    # 解析EMG数据 (8通道)
                    values = [float(x) for x in line.split(',')]
                    if len(values) >= 8:
                        return np.array(values[:8])
            except Exception as e:
                logger.error(f"EMG数据读取错误: {e}")
        return None

    def read_gsr_data(self) -> Optional[float]:
        """读取GSR数据"""
        if self.gsr_serial and self.gsr_serial.is_open:
            try:
                # 读取一行数据
                line = self.gsr_serial.readline().decode('utf-8').strip()
                if line:
                    return float(line)
            except Exception as e:
                logger.error(f"GSR数据读取错误: {e}")
        return None

    def data_acquisition_thread(self):
        """数据采集线程"""
        emg_counter = 0
        gsr_counter = 0
        emg_interval = 1.0 / self.config['emg_sample_rate']
        gsr_interval = 1.0 / self.config['gsr_sample_rate']
        last_emg_time = time.time()
        last_gsr_time = time.time()

        logger.info("启动数据采集线程")

        while self.running:
            current_time = time.time()

            # EMG数据采集 (1000Hz)
            if current_time - last_emg_time >= emg_interval:
                emg_data = self.read_emg_data()
                if emg_data is not None:
                    self.emg_data.append(emg_data)
                    emg_counter += 1
                last_emg_time = current_time

            # GSR数据采集 (100Hz)
            if current_time - last_gsr_time >= gsr_interval:
                gsr_data = self.read_gsr_data()
                if gsr_data is not None:
                    self.gsr_data.append(gsr_data)
                    self.timestamps.append(current_time)
                    gsr_counter += 1
                last_gsr_time = current_time

            # 控制循环频率
            time.sleep(0.0001)  # 0.1ms

        logger.info(f"采集线程结束: EMG={emg_counter}, GSR={gsr_counter}")

    def start_collection(self, label: str, duration: int = None):
        """开始采集指定标签的数据"""
        if self.collecting:
            logger.warning("正在采集中，请先停止当前采集")
            return False

        if duration is None:
            duration = self.config['collection_duration']

        self.current_label = label
        self.collection_start_time = time.time()
        self.collecting = True

        logger.info(f"开始采集 '{label}' 数据，时长: {duration}秒")
        return True

    def stop_collection(self):
        """停止当前采集"""
        if not self.collecting:
            return False

        self.collecting = False

        # 保存采集的数据
        self.save_collection_data()

        logger.info(f"停止采集 '{self.current_label}' 数据")
        self.current_label = None
        return True

    def save_collection_data(self):
        """保存采集的数据"""
        if self.current_label is None:
            return

        # 计算实际采集时间
        collection_duration = time.time() - self.collection_start_time

        # 准备数据
        emg_array = np.array(list(self.emg_data))
        gsr_array = np.array(list(self.gsr_data))
        timestamp_array = np.array(list(self.timestamps))

        if len(emg_array) == 0 or len(gsr_array) == 0:
            logger.warning("没有采集到数据")
            return

        # 对齐数据 (以GSR为基准)
        if len(emg_array) > len(gsr_array) * 10:  # EMG采样率是GSR的10倍
            emg_ratio = len(emg_array) // len(gsr_array)
            emg_downsampled = emg_array[:len(gsr_array) * emg_ratio]
            emg_reshaped = emg_downsampled.reshape(-1, emg_ratio, 8)
            emg_averaged = np.mean(emg_reshaped, axis=1)
        else:
            emg_averaged = emg_array[:len(gsr_array)]

        # 创建DataFrame
        data_dict = {
            'timestamp': timestamp_array[:len(gsr_array)],
            'emg1': emg_averaged[:, 0],
            'emg2': emg_averaged[:, 1],
            'emg3': emg_averaged[:, 2],
            'emg4': emg_averaged[:, 3],
            'emg5': emg_averaged[:, 4],
            'emg6': emg_averaged[:, 5],
            'emg7': emg_averaged[:, 6],
            'emg8': emg_averaged[:, 7],
            'gsr': gsr_array,
            'label': self.current_label,
            'session': self.config['session_name']
        }

        df = pd.DataFrame(data_dict)

        # 生成文件名
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['session_name']}_{self.current_label}_{timestamp_str}.csv"
        filepath = os.path.join(self.config['output_dir'], filename)

        # 保存数据
        df.to_csv(filepath, index=False)
        logger.info(f"数据已保存: {filepath} ({len(df)} 样本)")

        # 保存元数据
        metadata = {
            'session': self.config['session_name'],
            'label': self.current_label,
            'duration': collection_duration,
            'samples': len(df),
            'emg_sample_rate': self.config['emg_sample_rate'],
            'gsr_sample_rate': self.config['gsr_sample_rate'],
            'emg_port': self.config['emg_port'],
            'gsr_port': self.config['gsr_port'],
            'collection_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata_file = filepath.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def start(self):
        """启动数据采集器"""
        if self.running:
            logger.warning("采集器已在运行")
            return

        self.running = True

        # 启动数据采集线程
        acquisition_thread = threading.Thread(target=self.data_acquisition_thread)
        acquisition_thread.daemon = True
        self.threads.append(acquisition_thread)
        acquisition_thread.start()

        logger.info("数据采集器启动成功")

    def stop(self):
        """停止数据采集器"""
        logger.info("正在停止数据采集器...")

        # 停止当前采集
        if self.collecting:
            self.stop_collection()

        self.running = False

        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=2.0)

        # 关闭串口连接
        if self.emg_serial and self.emg_serial.is_open:
            self.emg_serial.close()
        if self.gsr_serial and self.gsr_serial.is_open:
            self.gsr_serial.close()

        logger.info("数据采集器已停止")

    def get_status(self) -> Dict:
        """获取采集器状态"""
        return {
            'running': self.running,
            'collecting': self.collecting,
            'current_label': self.current_label,
            'emg_queue_size': len(self.emg_data),
            'gsr_queue_size': len(self.gsr_data),
            'emg_connected': self.emg_serial and self.emg_serial.is_open,
            'gsr_connected': self.gsr_serial and self.gsr_serial.is_open,
            'collection_time': time.time() - self.collection_start_time if self.collection_start_time else 0
        }

def interactive_collection():
    """交互式数据采集"""
    print("=== EMG + GSR 数据采集工具 ===\n")

    # 创建采集器
    collector = DataCollector()
    collector.start()

    try:
        while True:
            print("\n=== 采集菜单 ===")
            print("1. 采集手势数据")
            print("2. 采集状态数据")
            print("3. 查看采集状态")
            print("4. 停止当前采集")
            print("5. 退出")

            choice = input("\n请选择操作 (1-5): ").strip()

            if choice == '1':
                print("\n=== 手势数据采集 ===")
                for i, label in enumerate(collector.config['gesture_labels'], 1):
                    print(f"{i}. {label}")

                try:
                    gesture_choice = int(input(f"请选择手势 (1-{len(collector.config['gesture_labels'])}): "))
                    if 1 <= gesture_choice <= len(collector.config['gesture_labels']):
                        label = collector.config['gesture_labels'][gesture_choice - 1]
                        duration = int(input("采集时长 (秒, 默认30): ") or "30")

                        collector.start_collection(label, duration)
                        print(f"正在采集 '{label}'，持续 {duration} 秒...")
                        time.sleep(duration)
                        collector.stop_collection()
                    else:
                        print("无效选择")
                except ValueError:
                    print("请输入有效数字")

            elif choice == '2':
                print("\n=== 状态数据采集 ===")
                for i, label in enumerate(collector.config['state_labels'], 1):
                    print(f"{i}. {label}")

                try:
                    state_choice = int(input(f"请选择状态 (1-{len(collector.config['state_labels'])}): "))
                    if 1 <= state_choice <= len(collector.config['state_labels']):
                        label = collector.config['state_labels'][state_choice - 1]
                        duration = int(input("采集时长 (秒, 默认30): ") or "30")

                        collector.start_collection(label, duration)
                        print(f"正在采集 '{label}'，持续 {duration} 秒...")
                        time.sleep(duration)
                        collector.stop_collection()
                    else:
                        print("无效选择")
                except ValueError:
                    print("请输入有效数字")

            elif choice == '3':
                status = collector.get_status()
                print(f"\n=== 采集器状态 ===")
                print(f"运行状态: {status['running']}")
                print(f"采集中: {status['collecting']}")
                print(f"当前标签: {status['current_label']}")
                print(f"EMG队列: {status['emg_queue_size']}")
                print(f"GSR队列: {status['gsr_queue_size']}")
                print(f"EMG连接: {status['emg_connected']}")
                print(f"GSR连接: {status['gsr_connected']}")
                if status['collection_time'] > 0:
                    print(f"已采集: {status['collection_time']:.1f} 秒")

            elif choice == '4':
                if collector.collecting:
                    collector.stop_collection()
                    print("当前采集已停止")
                else:
                    print("当前没有在采集")

            elif choice == '5':
                break

            else:
                print("无效选择，请重试")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        collector.stop()
        print("数据采集器已停止")

def main():
    """主函数"""
    interactive_collection()

if __name__ == "__main__":
    main()