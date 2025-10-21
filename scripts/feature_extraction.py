#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG + GSR 特征提取模块
基于LibEMG的信号处理和特征提取
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import kurtosis, skew
import logging

try:
    from libemg import SignalProcessor, FeatureExtractor, Windowing
    LIBEMG_AVAILABLE = True
except ImportError:
    LIBEMG_AVAILABLE = False
    logging.warning("LibEMG not available, using custom implementation")

# 自定义EMG处理实现
class CustomSignalProcessor:
    """自定义EMG信号处理器"""

    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate

    def bandpass_filter(self, data, low=20, high=450):
        """带通滤波器 (20-450Hz)"""
        nyquist = self.sample_rate / 2
        low_cut = low / nyquist
        high_cut = high / nyquist

        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        return signal.filtfilt(b, a, data)

class CustomFeatureExtractor:
    """自定义特征提取器"""

    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate

    def extract_rms(self, window_data):
        """均方根特征"""
        return np.sqrt(np.mean(window_data ** 2, axis=-1))

    def extract_mdf(self, window_data):
        """平均差分频率特征"""
        diffs = np.diff(window_data, axis=-1)
        return np.mean(np.abs(diffs), axis=-1)

    def extract_zc(self, window_data):
        """过零率特征"""
        zc_count = np.sum(np.diff(np.sign(window_data), axis=-1) != 0, axis=-1)
        return zc_count

    def extract_wl(self, window_data):
        """波形长度特征"""
        return np.sum(np.abs(np.diff(window_data, axis=-1)), axis=-1)

    def extract_features(self, window_data, methods=None):
        """提取多种特征"""
        if methods is None:
            methods = ['RMS', 'MDF', 'ZC', 'WL']

        features = []
        for method in methods:
            if method == 'RMS':
                features.append(self.extract_rms(window_data))
            elif method == 'MDF':
                features.append(self.extract_mdf(window_data))
            elif method == 'ZC':
                features.append(self.extract_zc(window_data))
            elif method == 'WL':
                features.append(self.extract_wl(window_data))

        return np.array(features)

class GSRFeatureExtractor:
    """GSR特征提取器"""

    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate

    def extract_mean(self, window_data):
        """均值特征"""
        return np.mean(window_data)

    def extract_std(self, window_data):
        """标准差特征"""
        return np.std(window_data)

    def extract_diff_mean(self, window_data):
        """差分均值特征"""
        diffs = np.diff(window_data)
        return np.mean(np.abs(diffs))

    def extract_peaks(self, window_data, prominence=0.1):
        """峰计数特征"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(window_data, prominence=prominence)
        return len(peaks)

    def extract_skewness(self, window_data):
        """偏度特征"""
        return skew(window_data)

    def extract_kurtosis(self, window_data):
        """峰度特征"""
        return kurtosis(window_data)

    def extract_features(self, window_data):
        """提取GSR特征"""
        features = []
        features.append(self.extract_mean(window_data))
        features.append(self.extract_std(window_data))
        features.append(self.extract_diff_mean(window_data))
        features.append(self.extract_peaks(window_data))
        features.append(self.extract_skewness(window_data))
        features.append(self.extract_kurtosis(window_data))

        return np.array(features)

class UnifiedFeatureExtractor:
    """统一的特征提取器 (EMG + GSR)"""

    def __init__(self, sample_rate_emg=1000, sample_rate_gsr=100):
        self.sample_rate_emg = sample_rate_emg
        self.sample_rate_gsr = sample_rate_gsr

        # 初始化EMG处理器
        if LIBEMG_AVAILABLE:
            self.emg_processor = SignalProcessor()
            self.emg_extractor = FeatureExtractor()
        else:
            self.emg_processor = CustomSignalProcessor(sample_rate_emg)
            self.emg_extractor = CustomFeatureExtractor(sample_rate_emg)

        # 初始化GSR处理器
        self.gsr_extractor = GSRFeatureExtractor(sample_rate_gsr)

        logging.info(f"Initialized feature extractor with sample rates: EMG={sample_rate_emg}Hz, GSR={sample_rate_gsr}Hz")

    def process_emg_signal(self, emg_data, low_freq=20, high_freq=450):
        """处理EMG信号"""
        # 带通滤波
        filtered_emg = self.emg_processor.bandpass_filter(emg_data, low_freq, high_freq,
                                                   fs=self.sample_rate_emg)
        return filtered_emg

    def create_windows(self, data, window_size=256, step_size=64):
        """创建滑动窗口"""
        num_windows = (len(data) - window_size) // step_size + 1
        windows = []

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            windows.append(data[start:end])

        return np.array(windows)

    def extract_emg_features(self, emg_windows, methods=None):
        """提取EMG特征"""
        if methods is None:
            methods = ['RMS', 'MDF', 'ZC', 'WL']

        features_list = []
        for window in emg_windows:
            features = self.emg_extractor.extract_features(window, methods)
            features_list.append(features)

        return np.array(features_list)

    def extract_gsr_features(self, gsr_windows):
        """提取GSR特征"""
        features_list = []
        for window in gsr_windows:
            features = self.gsr_extractor.extract_features(window)
            features_list.append(features)

        return np.array(features_list)

    def extract_combined_features(self, emg_data, gsr_data, emg_window_size=256,
                                emg_step_size=64, gsr_window_size=25, gsr_step_size=5):
        """提取组合特征 (EMG + GSR)"""

        # 处理EMG信号
        processed_emg = self.process_emg_signal(emg_data)
        emg_windows = self.create_windows(processed_emg, emg_window_size, emg_step_size)

        # 处理GSR信号 (降采样到EMG窗口大小)
        # 简单的降采样方法
        ratio = len(processed_emg) // len(gsr_data)
        if ratio > 1:
            # 对GSR数据进行插值
            gsr_resampled = np.interp(
                np.linspace(0, len(gsr_data)-1, len(processed_emg)),
                np.arange(len(gsr_data)),
                gsr_data
            )
        else:
            gsr_resampled = gsr_data

        gsr_windows = self.create_windows(gsr_resampled, gsr_window_size, gsr_step_size)

        # 调整窗口数量匹配
        min_windows = min(len(emg_windows), len(gsr_windows))
        emg_windows = emg_windows[:min_windows]
        gsr_windows = gsr_windows[:min_windows]

        # 提取特征
        emg_features = self.extract_emg_features(emg_windows)
        gsr_features = self.extract_gsr_features(gsr_windows)

        # 组合特征
        combined_features = np.concatenate([emg_features, gsr_features], axis=1)

        return combined_features, emg_windows, gsr_windows

def load_and_preprocess_data(file_path: str,
                           sample_rate_emg: int = 1000,
                           sample_rate_gsr: int = 100) -> Dict:
    """加载和预处理数据"""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}: {data.shape}")

        # 假设数据格式: timestamp, emg1, emg2, ..., emg8, gsr, label_gesture, label_state
        emg_columns = [col for col in data.columns if col.startswith('emg')]
        gsr_columns = [col for col in data.columns if col.startswith('gsr')]

        if not emg_columns:
            raise ValueError("No EMG columns found in data")
        if not gsr_columns:
            raise ValueError("No GSR columns found in data")

        # 提取EMG数据 (取平均或多通道处理)
        if len(emg_columns) > 1:
            emg_data = data[emg_columns].values
        else:
            emg_data = data[emg_columns[0]].values

        # 提取GSR数据
        gsr_data = data[gsr_columns[0]].values  # 假设单通道GSR

        # 提取标签
        labels = {
            'gesture': data['label_gesture'].values if 'label_gesture' in data.columns else None,
            'state': data['label_state'].values if 'label_state' in data.columns else None,
            'subject': data.get('subject_id', 'unknown').values
        }

        return {
            'emg_data': emg_data,
            'gsr_data': gsr_data,
            'labels': labels,
            'sample_rate_emg': sample_rate_emg,
            'sample_rate_gsr': sample_rate_gsr
        }

    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def extract_features_from_file(file_path: str,
                             extractor: UnifiedFeatureExtractor,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """从文件提取特征"""

    # 加载和预处理数据
    data_dict = load_and_preprocess_data(file_path, **kwargs)

    # 提取组合特征
    features, emg_windows, gsr_windows = extractor.extract_combined_features(
        data_dict['emg_data'],
        data_dict['gsr_data']
    )

    return features, data_dict['labels']

# 示例使用
if __name__ == "__main__":
    # 初始化特征提取器
    extractor = UnifiedFeatureExtractor()

    # 示例数据生成 (实际使用时替换为真实数据加载)
    sample_emg = np.random.randn(10000)  # 10秒的8通道EMG数据
    sample_gsr = np.random.randn(1000)   # 1秒的GSR数据

    # 添加一些真实信号特征
    t_emg = np.linspace(0, 10, 10000)
    sample_emg += 0.5 * np.sin(2 * np.pi * 50 * t_emg)  # 50Hz正弦波
    sample_emg += 0.2 * np.random.randn(10000)       # 噪声

    t_gsr = np.linspace(0, 1, 1000)
    sample_gsr += 0.1 * np.sin(2 * np.pi * 0.1 * t_gsr)  # 0.1Hz正弦波
    sample_gsr += 0.05 * np.random.randn(1000)          # 噪声

    # 提取特征
    features, emg_windows, gsr_windows = extractor.extract_combined_features(
        sample_emg, sample_gsr
    )

    print(f"特征矩阵形状: {features.shape}")
    print(f"EMG窗口数量: {len(emg_windows)}")
    print(f"GSR窗口数量: {len(gsr_windows)}")
    print(f"特征维度: {features.shape[1]}")