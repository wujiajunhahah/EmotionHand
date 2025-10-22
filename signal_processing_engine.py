#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 企业级信号处理引擎
Professional EMG+GSR Signal Processing Pipeline

基于专业预处理铁三角的实时信号处理系统:
- 信号→时间窗→归一化
- 干净、稳定、低延迟
- 质量监测、异常处理、个体化校准

Author: EmotionHand Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from collections import deque
from threading import Thread, Lock
import queue

# 信号处理库
from scipy.signal import iirnotch, butter, filtfilt, welch
from scipy.stats import iqr

logger = logging.getLogger(__name__)


class SignalQuality(NamedTuple):
    """信号质量指标"""
    snr: float  # 信噪比 (dB)
    clipping_rate: float  # 夹顶率 (0-1)
    artifact_rate: float  # 伪迹率 (0-1)
    connectivity: bool  # 连接状态
    quality_score: float  # 综合质量评分 (0-1)


class EMGFeatures(NamedTuple):
    """EMG特征集合"""
    rms: float  # 均方根 - 肌肉激活强度
    mdf: float  # 中位频 - 疲劳指标
    zc: int  # 过零率 - 紧张度
    wl: float  # 波长长度 - 不稳定性
    mav: float  # 平均绝对值
    ssi: float  # 平方和积分
    frequency_bands: Dict[str, float]  # 频带能量


class GSRFeatures(NamedTuple):
    """GSR特征集合"""
    tonic: float  # 基调水平 - 基础皮电
    phasic: float  # 反应性 - 瞬时变化
    scr_count: int  # 皮肤电导反应次数
    rise_time: float  # 上升时间
    amplitude: float  # 反应幅度
    derivative: float  # 变化速度


@dataclass
class CalibrationProfile:
    """个体化校准档案"""
    user_id: str
    timestamp: float

    # EMG基准值
    emg_baseline_rms: Dict[str, float]  # 各通道基准RMS
    emg_baseline_mdf: Dict[str, float]  # 各通道基准MDF

    # GSR基准值
    gsr_baseline_tonic: float
    gsr_scr_threshold: float

    # 归一化参数
    emg_quantiles: Dict[str, Dict[str, float]]  # p10, p90 for each channel
    gsr_quantiles: Dict[str, float]  # p10, p90 for GSR

    # 质量阈值
    snr_threshold: float
    clipping_threshold: float


class EMGProcessor:
    """EMG信号处理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.fs = config['emg']['sample_rate']
        self.window_size = config['window']['size']
        self.overlap_ratio = config['window']['overlap_ratio']

        # 预计算滤波器系数
        self._init_filters()

    def _init_filters(self):
        """初始化滤波器系数"""
        fs = self.fs

        # EMG带通滤波器 (20-450 Hz)
        low = 20 / (fs / 2)
        high = 450 / (fs / 2)
        self.b_band, self.a_band = butter(
            4, [low, high], btype='band'
        )

        # 工频陷波滤波器 (50 Hz)
        notch_freq = self.config['emg'].get('notch_freq', 50)
        Q = self.config['emg'].get('notch_q', 30)
        self.b_notch, self.a_notch = iirnotch(
            notch_freq / (fs / 2), Q
        )

        # 次谐波陷波 (100 Hz)
        self.b_notch2, self.a_notch2 = iirnotch(
            (notch_freq * 2) / (fs / 2), Q
        )

    def filter_emg(self, emg_signal: np.ndarray) -> np.ndarray:
        """EMG滤波处理"""
        # 带通滤波
        filtered = filtfilt(self.b_band, self.a_band, emg_signal)

        # 工频陷波
        filtered = filtfilt(self.b_notch, self.a_notch, filtered)

        # 次谐波陷波
        filtered = filtfilt(self.b_notch2, self.a_notch2, filtered)

        # 去直流
        filtered = filtered - np.mean(filtered)

        return filtered

    def extract_features(self, emg_filtered: np.ndarray) -> EMGFeatures:
        """提取EMG特征"""
        # 整流
        rectified = np.abs(emg_filtered)

        # 包络提取 (5-10 Hz低通)
        envelope_fs = self.config['emg'].get('envelope_freq', 8)
        envelope_low = envelope_fs / (self.fs / 2)
        b_env, a_env = butter(2, envelope_low, btype='low')
        envelope = filtfilt(b_env, a_env, rectified)

        # 基础特征
        rms = float(np.sqrt(np.mean(emg_filtered ** 2)))
        mav = float(np.mean(rectified))
        ssi = float(np.sum(emg_filtered ** 2))

        # 时域特征
        zc = int(np.sum(np.diff(np.sign(emg_filtered)) != 0))
        wl = float(np.sum(np.abs(np.diff(emg_filtered))))

        # 频域特征 - MDF
        mdf = self._calculate_mdf(emg_filtered)

        # 频带能量
        frequency_bands = self._calculate_frequency_bands(emg_filtered)

        return EMGFeatures(
            rms=rms,
            mdf=mdf,
            zc=zc,
            wl=wl,
            mav=mav,
            ssi=ssi,
            frequency_bands=frequency_bands
        )

    def _calculate_mdf(self, signal: np.ndarray) -> float:
        """计算中位频率 (Median Frequency)"""
        try:
            f, P = welch(signal, fs=self.fs, nperseg=256, noverlap=128)
            c = np.cumsum(P) / np.sum(P)
            return float(f[np.searchsorted(c, 0.5)])
        except:
            return 0.0

    def _calculate_frequency_bands(self, signal: np.ndarray) -> Dict[str, float]:
        """计算频带能量"""
        try:
            f, P = welch(signal, fs=self.fs, nperseg=256, noverlap=128)

            bands = {}
            # 频带定义
            band_ranges = {
                'alpha': [8, 12],
                'beta': [13, 30],
                'gamma': [31, 100],
                'high_gamma': [101, 200]
            }

            for band_name, (f_low, f_high) in band_ranges.items():
                mask = (f >= f_low) & (f <= f_high)
                bands[band_name] = float(np.sum(P[mask]))

            return bands
        except:
            return {'alpha': 0, 'beta': 0, 'gamma': 0, 'high_gamma': 0}

    def assess_quality(self, emg_raw: np.ndarray, emg_filtered: np.ndarray) -> SignalQuality:
        """评估EMG信号质量"""
        # 夹顶检测
        adc_bits = self.config['emg'].get('adc_bits', 12)
        max_val = (2 ** (adc_bits - 1)) - 1

        clipping_samples = np.sum(
            (np.abs(emg_raw) >= max_val * 0.98) |
            (np.abs(emg_raw) <= max_val * 0.02)
        )
        clipping_rate = clipping_samples / len(emg_raw)

        # SNR估计
        signal_power = np.var(emg_filtered)
        noise_power = np.var(emg_raw - emg_filtered)
        snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-10))

        # 运动伪迹检测 (5σ异常)
        diff_signal = np.diff(emg_filtered)
        sigma = np.std(diff_signal)
        artifacts = np.sum(np.abs(diff_signal) > 5 * sigma)
        artifact_rate = artifacts / len(diff_signal)

        # 连接状态
        connectivity = not (np.max(np.abs(emg_raw)) < max_val * 0.01)

        # 综合质量评分
        quality_score = (
            (1 - clipping_rate) * 0.4 +
            min(snr_db / 20, 1) * 0.3 +
            (1 - artifact_rate) * 0.2 +
            (1 if connectivity else 0) * 0.1
        )

        return SignalQuality(
            snr=snr_db,
            clipping_rate=clipping_rate,
            artifact_rate=artifact_rate,
            connectivity=connectivity,
            quality_score=max(0, min(1, quality_score))
        )


class GSRProcessor:
    """GSR信号处理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.fs = config['gsr']['sample_rate']

        # 存储历史数据用于分析
        self.gsr_history = deque(maxlen=60 * self.fs)  # 1分钟历史

    def filter_gsr(self, gsr_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GSR滤波处理"""
        # 去漂移 - Tonic提取 (0.05-1.0 Hz)
        tonic_low = self.config['gsr'].get('tonic_cutoff', 0.5) / (self.fs / 2)
        b_tonic, a_tonic = butter(2, tonic_low, btype='low')
        tonic = filtfilt(b_tonic, a_tonic, gsr_signal)

        # Phasic提取 - 去除慢漂移
        phasic_high = self.config['gsr'].get('phasic_cutoff', 0.05) / (self.fs / 2)
        b_phasic, a_phasic = butter(2, phasic_high, btype='high')
        phasic = filtfilt(b_phasic, a_phasic, gsr_signal)

        return tonic, phasic

    def extract_features(self, gsr_raw: np.ndarray, tonic: np.ndarray, phasic: np.ndarray) -> GSRFeatures:
        """提取GSR特征"""
        # 基调特征
        tonic_level = float(np.mean(tonic))

        # 反应性特征
        phasic_level = float(np.mean(np.abs(phasic)))

        # 变化速度
        derivative = float(np.mean(np.abs(np.diff(gsr_raw))))

        # SCR检测
        scr_peaks = self._detect_scr_peaks(phasic)
        scr_count = len(scr_peaks)

        # 幅度和上升时间
        amplitudes = []
        rise_times = []
        for peak_idx in scr_peaks:
            # 简化的幅度计算
            amplitude = float(peak_idx)
            amplitudes.append(amplitude)

        avg_amplitude = np.mean(amplitudes) if amplitudes else 0.0
        avg_rise_time = np.mean(rise_times) if rise_times else 0.0

        return GSRFeatures(
            tonic=tonic_level,
            phasic=phasic_level,
            scr_count=scr_count,
            rise_time=avg_rise_time,
            amplitude=avg_amplitude,
            derivative=derivative
        )

    def _detect_scr_peaks(self, phasic_signal: np.ndarray) -> List[int]:
        """检测SCR峰值"""
        threshold = self.config['gsr'].get('scr_threshold', 0.03)

        # 简化的峰值检测
        peaks = []
        for i in range(1, len(phasic_signal) - 1):
            if (phasic_signal[i] > threshold and
                phasic_signal[i] > phasic_signal[i-1] and
                phasic_signal[i] > phasic_signal[i+1]):
                peaks.append(i)

        return peaks

    def assess_quality(self, gsr_raw: np.ndarray) -> SignalQuality:
        """评估GSR信号质量"""
        # 接触检测 (瞬间跌落到0附近)
        min_val = np.min(gsr_raw)
        connectivity = not (min_val < 0.01 * np.max(gsr_raw))

        # 变异性检测
        cv = np.std(gsr_raw) / max(np.mean(gsr_raw), 1e-10)
        variability_score = min(cv / 0.5, 1)  # 标准化到0-1

        # 稳定性检测
        recent_diff = np.mean(np.abs(np.diff(gsr_raw[-min(100, len(gsr_raw)):])))
        stability_score = 1 - min(recent_diff / 0.1, 1)

        # 综合质量评分
        quality_score = (
            (1 if connectivity else 0) * 0.4 +
            variability_score * 0.3 +
            stability_score * 0.3
        )

        return SignalQuality(
            snr=0.0,  # GSR不计算SNR
            clipping_rate=0.0,  # GSR通常不夹顶
            artifact_rate=0.0,
            connectivity=connectivity,
            quality_score=max(0, min(1, quality_score))
        )


class PersonalizedNormalizer:
    """个体化归一化处理器"""

    def __init__(self):
        self.calibration_profile: Optional[CalibrationProfile] = None

    def load_calibration(self, profile_path: str) -> bool:
        """加载校准档案"""
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)

            self.calibration_profile = CalibrationProfile(**data)
            logger.info(f"已加载校准档案: {profile_path}")
            return True
        except Exception as e:
            logger.error(f"加载校准档案失败: {e}")
            return False

    def normalize_features(self, emg_features: EMGFeatures, gsr_features: GSRFeatures) -> Dict[str, float]:
        """归一化特征到[0,1]范围"""
        if self.calibration_profile is None:
            # 无校准档案时使用简单归一化
            return self._simple_normalize(emg_features, gsr_features)

        normalized = {}

        # EMG特征归一化
        profile = self.calibration_profile

        # RMS归一化
        emg_q = profile.emg_quantiles.get('rms', {'p10': 0.1, 'p90': 1.0})
        normalized['rms'] = self._qnorm(emg_features.rms, emg_q['p10'], emg_q['p90'])

        # MDF归一化
        mdf_q = profile.emg_quantiles.get('mdf', {'p10': 50, 'p90': 150})
        normalized['mdf'] = self._qnorm(emg_features.mdf, mdf_q['p10'], mdf_q['p90'])

        # GSR特征归一化
        gsr_q = profile.gsr_quantiles
        normalized['gsr_tonic'] = self._qnorm(gsr_features.tonic, gsr_q['p10'], gsr_q['p90'])
        normalized['gsr_phasic'] = self._qnorm(gsr_features.phasic, gsr_q['p10'], gsr_q['p90'])

        return normalized

    def _simple_normalize(self, emg_features: EMGFeatures, gsr_features: GSRFeatures) -> Dict[str, float]:
        """简单归一化 (无校准档案时)"""
        return {
            'rms': min(emg_features.rms / 2.0, 1.0),
            'mdf': min(emg_features.mdf / 200.0, 1.0),
            'gsr_tonic': min(gsr_features.tonic / 1.0, 1.0),
            'gsr_phasic': min(gsr_features.phasic / 0.5, 1.0)
        }

    def _qnorm(self, x: float, p10: float, p90: float) -> float:
        """分位归一化"""
        return np.clip((x - p10) / max(p90 - p10, 1e-6), 0, 1)


class RealTimeSignalProcessor:
    """实时信号处理引擎"""

    def __init__(self, config_path: str = 'signal_processing_config.json'):
        self.config = self._load_config(config_path)

        # 初始化处理器
        self.emg_processor = EMGProcessor(self.config)
        self.gsr_processor = GSRProcessor(self.config)
        self.normalizer = PersonalizedNormalizer()

        # 数据缓冲区
        self.emg_buffer = deque(maxlen=self.config['window']['size'])
        self.gsr_buffer = deque(maxlen=self.config['window']['size'])

        # 质量监测
        self.quality_history = deque(maxlen=100)

        # 性能监控
        self.processing_times = deque(maxlen=100)

        # 线程安全
        self.data_lock = Lock()
        self.running = False

        logger.info("信号处理引擎初始化完成")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置"""
        default_config = {
            "emg": {
                "sample_rate": 1000,
                "notch_freq": 50,
                "notch_q": 30,
                "envelope_freq": 8,
                "adc_bits": 12
            },
            "gsr": {
                "sample_rate": 100,
                "tonic_cutoff": 0.5,
                "phasic_cutoff": 0.05,
                "scr_threshold": 0.03
            },
            "window": {
                "size": 256,
                "overlap_ratio": 0.75
            },
            "quality": {
                "min_snr": 6.0,
                "max_clipping_rate": 0.01,
                "min_quality_score": 0.7
            }
        }

        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # 合并配置
                for key in default_config:
                    if key in user_config:
                        default_config[key].update(user_config[key])
                logger.info(f"已加载配置: {config_path}")
            except Exception as e:
                logger.warning(f"配置加载失败，使用默认配置: {e}")

        return default_config

    def start(self):
        """启动处理引擎"""
        self.running = True
        logger.info("信号处理引擎已启动")

    def stop(self):
        """停止处理引擎"""
        self.running = False
        logger.info("信号处理引擎已停止")

    def add_data(self, emg_sample: List[float], gsr_sample: float, timestamp: float = None):
        """添加新数据样本"""
        if timestamp is None:
            timestamp = time.time()

        with self.data_lock:
            self.emg_buffer.extend(emg_sample)
            self.gsr_buffer.append(gsr_sample)

    def process_window(self) -> Optional[Dict]:
        """处理当前时间窗"""
        if len(self.emg_buffer) < self.config['window']['size']:
            return None

        start_time = time.time()

        with self.data_lock:
            # 获取窗口数据
            window_emg = np.array(list(self.emg_buffer)[-self.config['window']['size']:])
            window_gsr = np.array(list(self.gsr_buffer)[-self.config['window']['size']:])

        try:
            # EMG处理
            emg_filtered = self.emg_processor.filter_emg(window_emg)
            emg_features = self.emg_processor.extract_features(emg_filtered)
            emg_quality = self.emg_processor.assess_quality(window_emg, emg_filtered)

            # GSR处理
            gsr_tonic, gsr_phasic = self.gsr_processor.filter_gsr(window_gsr)
            gsr_features = self.gsr_processor.extract_features(window_gsr, gsr_tonic, gsr_phasic)
            gsr_quality = self.gsr_processor.assess_quality(window_gsr)

            # 归一化
            normalized_features = self.normalizer.normalize_features(emg_features, gsr_features)

            # 质量评估
            overall_quality = (emg_quality.quality_score + gsr_quality.quality_score) / 2
            self.quality_history.append(overall_quality)

            # 性能监控
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return {
                'timestamp': time.time(),
                'emg_features': emg_features._asdict(),
                'gsr_features': gsr_features._asdict(),
                'normalized_features': normalized_features,
                'quality': {
                    'emg': emg_quality._asdict(),
                    'gsr': gsr_quality._asdict(),
                    'overall': overall_quality
                },
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"处理窗口失败: {e}")
            return None

    def get_quality_status(self) -> Dict:
        """获取信号质量状态"""
        if not self.quality_history:
            return {'status': 'no_data', 'score': 0.0}

        avg_quality = np.mean(list(self.quality_history))

        if avg_quality >= self.config['quality']['min_quality_score']:
            status = 'excellent'
        elif avg_quality >= 0.5:
            status = 'good'
        elif avg_quality >= 0.3:
            status = 'poor'
        else:
            status = 'bad'

        return {
            'status': status,
            'score': avg_quality,
            'recent_quality': list(self.quality_history)[-10:]
        }

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.processing_times:
            return {'avg_time': 0, 'max_time': 0, 'fps': 0}

        times = list(self.processing_times)
        avg_time = np.mean(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': np.min(times),
            'fps': fps,
            'latency_ms': avg_time * 1000
        }


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    processor = RealTimeSignalProcessor()
    processor.start()

    # 模拟数据测试
    print("🧪 测试信号处理引擎...")

    for i in range(10):
        # 添加模拟数据
        emg_data = [np.random.randn() * 0.5 for _ in range(8)]
        gsr_data = 0.2 + np.random.randn() * 0.05

        processor.add_data(emg_data, gsr_data)

        # 处理
        if i >= 3:  # 等待缓冲区满
            result = processor.process_window()
            if result:
                print(f"窗口 {i}: 质量={result['quality']['overall']:.2f}, "
                      f"处理时间={result['processing_time']*1000:.1f}ms")

        time.sleep(0.1)

    # 显示统计
    quality_status = processor.get_quality_status()
    performance_stats = processor.get_performance_stats()

    print(f"\n📊 质量状态: {quality_status['status']} (评分: {quality_status['score']:.2f})")
    print(f"⚡ 性能统计: {performance_stats['fps']:.1f} FPS, "
          f"延迟: {performance_stats.get('latency_ms', performance_stats['avg_time']*1000):.1f}ms")

    processor.stop()