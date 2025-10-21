#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRT实时推理管线
EMG + GSR 特征提取 → 实时分类 → Unity 3D可视化
延迟 <100ms 的高性能管线
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import queue
import socket
import struct
import serial
import serial.tools.list_ports
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging

import joblib
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# 导入自定义模块
from feature_extraction import UnifiedFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimePipeline:
    """GRT风格的实时推理管线"""

    def __init__(self, config: Dict = None):
        # 默认配置
        self.config = {
            # 串口配置
            'emg_port': '/dev/tty.usbmodem1',  # Muscle Sensor v3
            'gsr_port': '/dev/tty.usbmodem2',  # GSR传感器
            'emg_baudrate': 115200,
            'gsr_baudrate': 9600,

            # 信号处理参数
            'emg_sample_rate': 1000,
            'gsr_sample_rate': 100,
            'emg_window_size': 256,
            'emg_step_size': 64,
            'gsr_window_size': 25,
            'gsr_step_size': 5,
            'emg_freq_range': [20, 450],  # Hz

            # 实时推理参数
            'prediction_threshold': 0.6,  # 置信度阈值
            'smoothing_window': 5,        # 预测平滑窗口
            'rejection_enabled': True,    # 拒识机制
            'max_latency': 100,          # 最大延迟(ms)

            # Unity通信
            'unity_ip': '127.0.0.1',
            'unity_port': 9001,
            'send_frequency': 50,        # Hz

            # 模型路径
            'gesture_model_path': './models/gesture_lightgbm.joblib',
            'state_model_path': './models/state_lightgbm.joblib',
            'scaler_path': './models/scaler.joblib',
            'label_encoder_path': './models/label_encoder.joblib'
        }

        if config:
            self.config.update(config)

        # 初始化组件
        self.feature_extractor = UnifiedFeatureExtractor(
            sample_rate_emg=self.config['emg_sample_rate'],
            sample_rate_gsr=self.config['gsr_sample_rate']
        )

        # 数据队列
        self.emg_queue = deque(maxlen=2000)  # 2秒的EMG数据
        self.gsr_queue = deque(maxlen=200)   # 2秒的GSR数据
        self.prediction_queue = deque(maxlen=self.config['smoothing_window'])

        # 实时统计
        self.stats = {
            'processed_samples': 0,
            'predictions_made': 0,
            'rejected_predictions': 0,
            'avg_latency': 0.0,
            'last_prediction_time': 0,
            'fps': 0.0
        }

        # 线程控制
        self.running = False
        self.threads = []

        # 加载模型
        self.load_models()

        # 初始化通信
        self.init_connections()

        logger.info("实时推理管线初始化完成")

    def load_models(self):
        """加载训练好的模型"""
        try:
            # 加载手势分类器
            if os.path.exists(self.config['gesture_model_path']):
                self.gesture_model = joblib.load(self.config['gesture_model_path'])
                logger.info("手势分类器加载成功")
            else:
                logger.warning(f"手势模型文件不存在: {self.config['gesture_model_path']}")
                self.gesture_model = None

            # 加载状态分类器
            if os.path.exists(self.config['state_model_path']):
                self.state_model = joblib.load(self.config['state_model_path'])
                logger.info("状态分类器加载成功")
            else:
                logger.warning(f"状态模型文件不存在: {self.config['state_model_path']}")
                self.state_model = None

            # 加载标准化器
            if os.path.exists(self.config['scaler_path']):
                self.scaler = joblib.load(self.config['scaler_path'])
            else:
                self.scaler = StandardScaler()
                logger.info("使用新的标准化器")

            # 加载标签编码器
            if os.path.exists(self.config['label_encoder_path']):
                self.label_encoders = joblib.load(self.config['label_encoder_path'])
            else:
                self.label_encoders = {'gesture': None, 'state': None}
                logger.info("使用新的标签编码器")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.gesture_model = None
            self.state_model = None
            self.scaler = StandardScaler()
            self.label_encoders = {'gesture': None, 'state': None}

    def init_connections(self):
        """初始化通信连接"""
        # 串口连接
        self.emg_serial = None
        self.gsr_serial = None

        # Unity UDP连接
        self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 尝试自动检测串口
        self.detect_serial_ports()

    def detect_serial_ports(self):
        """自动检测可用串口"""
        ports = serial.tools.list_ports.comports()
        emg_candidates = []
        gsr_candidates = []

        for port in ports:
            if 'usbmodem' in port.device.lower() or 'acm' in port.device.lower():
                if port.vid is not None:  # 有VID PID的设备
                    emg_candidates.append(port.device)
                else:
                    gsr_candidates.append(port.device)

        # 分配串口
        if emg_candidates and not self.emg_serial:
            try:
                self.emg_serial = serial.Serial(emg_candidates[0], self.config['emg_baudrate'])
                logger.info(f"EMG串口连接: {emg_candidates[0]}")
            except Exception as e:
                logger.error(f"EMG串口连接失败: {e}")

        if gsr_candidates and not self.gsr_serial:
            try:
                self.gsr_serial = serial.Serial(gsr_candidates[0], self.config['gsr_baudrate'])
                logger.info(f"GSR串口连接: {gsr_candidates[0]}")
            except Exception as e:
                logger.error(f"GSR串口连接失败: {e}")

    def read_emg_data(self) -> Optional[np.ndarray]:
        """读取EMG数据"""
        if self.emg_serial and self.emg_serial.is_open:
            try:
                # 读取一行数据
                line = self.emg_serial.readline().decode('utf-8').strip()
                if line:
                    # 假设数据格式: "emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8"
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

    def preprocess_signal(self, signal: np.ndarray, signal_type: str = 'emg') -> np.ndarray:
        """信号预处理"""
        if signal_type == 'emg':
            # 带通滤波
            nyquist = self.config['emg_sample_rate'] / 2
            low = self.config['emg_freq_range'][0] / nyquist
            high = self.config['emg_freq_range'][1] / nyquist

            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            return filtered
        else:
            # GSR信号处理 (低通滤波)
            nyquist = self.config['gsr_sample_rate'] / 2
            cutoff = 0.5 / nyquist

            b, a = butter(2, cutoff, btype='low')
            filtered = filtfilt(b, a, signal)
            return filtered

    def extract_real_time_features(self) -> Optional[np.ndarray]:
        """实时特征提取"""
        if len(self.emg_queue) < self.config['emg_window_size'] or \
           len(self.gsr_queue) < self.config['gsr_window_size']:
            return None

        try:
            # 获取最新窗口数据
            emg_data = np.array(list(self.emg_queue)[-self.config['emg_window_size']:])
            gsr_data = np.array(list(self.gsr_queue)[-self.config['gsr_window_size']:])

            # 预处理
            if len(emg_data.shape) == 1:
                emg_data = np.expand_dims(emg_data, axis=1)

            # 提取组合特征
            features, _, _ = self.feature_extractor.extract_combined_features(
                emg_data, gsr_data,
                self.config['emg_window_size'],
                self.config['emg_step_size'],
                self.config['gsr_window_size'],
                self.config['gsr_step_size']
            )

            # 标准化
            if features.shape[0] > 0:
                features = self.scaler.transform(features)
                return features[-1]  # 返回最新特征

        except Exception as e:
            logger.error(f"特征提取错误: {e}")

        return None

    def predict_with_confidence(self, features: np.ndarray, model) -> Tuple[str, float]:
        """带置信度的预测"""
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([features])[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]

                # 解码标签
                label_encoder = self.label_encoders.get('gesture')
                if label_encoder:
                    class_name = label_encoder.inverse_transform([predicted_class])[0]
                else:
                    class_name = f"class_{predicted_class}"

                return class_name, confidence
            else:
                # 不支持概率的模型
                prediction = model.predict([features])[0]
                return prediction, 1.0

        except Exception as e:
            logger.error(f"预测错误: {e}")
            return "Unknown", 0.0

    def smooth_predictions(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """预测平滑处理"""
        self.prediction_queue.append((prediction, confidence))

        if len(self.prediction_queue) < self.config['smoothing_window']:
            return prediction, confidence

        # 投票机制
        predictions = [p[0] for p in self.prediction_queue]
        confidences = [p[1] for p in self.prediction_queue]

        # 统计每个类别的出现次数和平均置信度
        class_counts = {}
        class_confidences = {}

        for pred, conf in zip(predictions, confidences):
            if pred not in class_counts:
                class_counts[pred] = 0
                class_confidences[pred] = []
            class_counts[pred] += 1
            class_confidences[pred].append(conf)

        # 选择出现次数最多的类别
        max_count = max(class_counts.values())
        candidates = [cls for cls, count in class_counts.items() if count == max_count]

        # 如果有多个候选，选择平均置信度最高的
        if len(candidates) > 1:
            best_class = max(candidates, key=lambda x: np.mean(class_confidences[x]))
        else:
            best_class = candidates[0]

        avg_confidence = np.mean(class_confidences[best_class])
        return best_class, avg_confidence

    def send_to_unity(self, gesture: str, state: str, confidence: float,
                     features: np.ndarray, latency: float):
        """发送预测结果到Unity"""
        try:
            # 构造数据包
            data = {
                'timestamp': time.time(),
                'gesture': gesture,
                'state': state,
                'confidence': confidence,
                'latency': latency,
                'feature_dim': len(features),
                'features_sample': features[:5].tolist() if len(features) > 5 else features.tolist()
            }

            # 转换为字符串发送
            message = f"{gesture}|{state}|{confidence:.3f}|{latency:.1f}"
            self.unity_socket.sendto(message.encode('utf-8'),
                                   (self.config['unity_ip'], self.config['unity_port']))

        except Exception as e:
            logger.error(f"Unity通信错误: {e}")

    def data_acquisition_thread(self):
        """数据采集线程"""
        logger.info("启动数据采集线程")

        while self.running:
            start_time = time.time()

            # 读取EMG数据
            emg_data = self.read_emg_data()
            if emg_data is not None:
                self.emg_queue.extend(emg_data)

            # 读取GSR数据
            gsr_data = self.read_gsr_data()
            if gsr_data is not None:
                self.gsr_queue.append(gsr_data)

            # 控制采样率
            elapsed = (time.time() - start_time) * 1000
            if elapsed < 1.0:  # 1ms采样间隔
                time.sleep(0.001)

    def inference_thread(self):
        """推理线程"""
        logger.info("启动推理线程")

        last_send_time = time.time()
        send_interval = 1.0 / self.config['send_frequency']

        while self.running:
            start_time = time.time()

            # 特征提取
            features = self.extract_real_time_features()
            if features is not None:
                # 手势预测
                if self.gesture_model:
                    gesture, gesture_conf = self.predict_with_confidence(features, self.gesture_model)
                    gesture, gesture_conf = self.smooth_predictions(gesture, gesture_conf)
                else:
                    gesture, gesture_conf = "NoModel", 0.0

                # 状态预测
                if self.state_model:
                    state, state_conf = self.predict_with_confidence(features, self.state_model)
                    state, state_conf = self.smooth_predictions(state, state_conf)
                else:
                    state, state_conf = "NoModel", 0.0

                # 拒识机制
                final_confidence = min(gesture_conf, state_conf)
                if final_confidence < self.config['prediction_threshold'] and self.config['rejection_enabled']:
                    gesture = "Neutral"
                    state = "Neutral"
                    final_confidence = 0.5
                    self.stats['rejected_predictions'] += 1

                # 计算延迟
                latency = (time.time() - start_time) * 1000

                # 更新统计
                self.stats['predictions_made'] += 1
                self.stats['avg_latency'] = (self.stats['avg_latency'] * 0.9 + latency * 0.1)
                self.stats['last_prediction_time'] = time.time()

                # 发送到Unity (控制发送频率)
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    self.send_to_unity(gesture, state, final_confidence, features, latency)
                    last_send_time = current_time

                # 更新FPS
                if elapsed > 0:
                    self.stats['fps'] = 1000.0 / elapsed

            # 控制推理频率
            elapsed = (time.time() - start_time) * 1000
            if elapsed < 10.0:  # 10ms推理间隔
                time.sleep(0.01)

    def start(self):
        """启动实时管线"""
        if self.running:
            logger.warning("管线已在运行")
            return

        self.running = True

        # 启动数据采集线程
        acquisition_thread = threading.Thread(target=self.data_acquisition_thread)
        acquisition_thread.daemon = True
        self.threads.append(acquisition_thread)
        acquisition_thread.start()

        # 启动推理线程
        inference_thread = threading.Thread(target=self.inference_thread)
        inference_thread.daemon = True
        self.threads.append(inference_thread)
        inference_thread.start()

        logger.info("实时推理管线启动成功")

    def stop(self):
        """停止管线"""
        logger.info("正在停止实时推理管线...")
        self.running = False

        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=2.0)

        # 关闭连接
        if self.emg_serial and self.emg_serial.is_open:
            self.emg_serial.close()
        if self.gsr_serial and self.gsr_serial.is_open:
            self.gsr_serial.close()
        if self.unity_socket:
            self.unity_socket.close()

        logger.info("实时推理管线已停止")

    def get_status(self) -> Dict:
        """获取管线状态"""
        return {
            'running': self.running,
            'emg_queue_size': len(self.emg_queue),
            'gsr_queue_size': len(self.gsr_queue),
            'prediction_queue_size': len(self.prediction_queue),
            'emg_connected': self.emg_serial and self.emg_serial.is_open,
            'gsr_connected': self.gsr_serial and self.gsr_serial.is_open,
            'stats': self.stats.copy()
        }

def main():
    """主函数"""
    # 配置管线
    config = {
        'emg_port': '/dev/tty.usbmodem1',
        'gsr_port': '/dev/tty.usbmodem2',
        'unity_ip': '127.0.0.1',
        'unity_port': 9001,
        'prediction_threshold': 0.6,
        'send_frequency': 50
    }

    # 创建管线
    pipeline = RealTimePipeline(config)

    try:
        # 启动管线
        pipeline.start()

        # 状态监控
        while True:
            status = pipeline.get_status()
            print(f"\n=== 管线状态 ===")
            print(f"运行状态: {status['running']}")
            print(f"EMG队列: {status['emg_queue_size']}/2000")
            print(f"GSR队列: {status['gsr_queue_size']}/200")
            print(f"预测队列: {status['prediction_queue_size']}/5")
            print(f"EMG连接: {status['emg_connected']}")
            print(f"GSR连接: {status['gsr_connected']}")
            print(f"预测次数: {status['stats']['predictions_made']}")
            print(f"拒识次数: {status['stats']['rejected_predictions']}")
            print(f"平均延迟: {status['stats']['avg_latency']:.1f}ms")
            print(f"FPS: {status['stats']['fps']:.1f}")

            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n用户中断，正在停止管线...")
    except Exception as e:
        logger.error(f"管线运行错误: {e}")
    finally:
        pipeline.stop()
        print("管线已停止")

if __name__ == "__main__":
    main()