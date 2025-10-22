#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 个体化校准系统
Personalized Calibration System

实现60秒快速校准，建立个体化档案：
- 30秒静息基准采集
- 30秒轻握活动采集
- 自动计算归一化参数
- 保存和加载校准档案

Author: EmotionHand Team
Version: 1.0.0
"""

import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import threading

from signal_processing_engine import (
    EMGProcessor, GSRProcessor, CalibrationProfile,
    EMGFeatures, GSRFeatures, SignalQuality
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSession:
    """校准会话数据"""
    user_id: str
    start_time: float
    phase: str  # 'rest', 'activity', 'completed'

    # 数据收集
    emg_rest_data: List[np.ndarray]
    emg_activity_data: List[np.ndarray]
    gsr_rest_data: List[float]
    gsr_activity_data: List[float]

    # 时间戳
    phase_start_time: float
    phase_duration: float

    # 结果
    calibration_profile: Optional[CalibrationProfile] = None


class CalibrationGuide:
    """校准过程引导器"""

    def __init__(self):
        self.messages = {
            'welcome': """
🎯 EmotionHand 个体化校准开始
=============================

校准过程将帮助系统了解您的生理基线：
• 建立个性化的EMG+GSR基准值
• 提高情绪状态识别准确性
• 适配您的肌肉活动和皮电特性

校准总时长: 60秒
请确保传感器已正确连接并佩戴舒适

按回车键开始...
            """,

            'rest_intro': """
🧘 第一阶段：静息基准采集 (30秒)
============================

请保持完全放松状态：
• 自然坐直，双手放在腿上
• 不要握拳或用力
• 保持正常呼吸
• 避免紧张或大幅动作

系统正在收集您的静息基线数据...
            """,

            'activity_intro': """
💪 第二阶段：轻握活动采集 (30秒)
==============================

请进行轻度手部活动：
• 轻轻握拳，保持2秒，然后放松
• 重复5-10次即可，无需用力
• 保持节奏平稳
• 动作幅度要小而一致

系统正在收集您的活动数据...
            """,

            'countdown': "剩余时间: {seconds}秒",
            'completed': """
✅ 校准完成！
===============

您的个人生理档案已建立：
• 静息基线: {rest_score:.1f}% 稳定度
• 活动基线: {activity_score:.1f}% 稳定度
• 整体质量: {overall_score:.1f}% 稳定度

校准档案已保存为: {profile_path}
现在可以开始实时情绪监测了！

            """,

            'error': """
❌ 校准失败
===========

原因: {error_message}

建议检查：
• 传感器连接是否牢固
• 电极是否正确贴附
• 环境干扰是否过大

请调整后重新尝试校准。
            """
        }

    def display_message(self, message_type: str, **kwargs):
        """显示引导信息"""
        message = self.messages.get(message_type, "")
        try:
            formatted_message = message.format(**kwargs)
        except:
            formatted_message = message

        print(formatted_message)


class CalibrationSystem:
    """个体化校准系统"""

    def __init__(self, config: Dict):
        self.config = config
        self.guide = CalibrationGuide()
        self.current_session: Optional[CalibrationSession] = None

        # 信号处理器
        self.emg_processor = EMGProcessor(config)
        self.gsr_processor = GSRProcessor(config)

        # 数据收集缓冲区
        self.data_buffer = deque(maxlen=1000)
        self.quality_buffer = deque(maxlen=50)

        # 线程控制
        self.calibrating = False
        self.calibration_thread: Optional[threading.Thread] = None

        logger.info("校准系统初始化完成")

    def start_calibration(self, user_id: str = "default_user") -> bool:
        """开始校准会话"""
        if self.calibrating:
            logger.warning("校准已在进行中")
            return False

        # 初始化会话
        self.current_session = CalibrationSession(
            user_id=user_id,
            start_time=time.time(),
            phase='welcome',
            emg_rest_data=[],
            emg_activity_data=[],
            gsr_rest_data=[],
            gsr_activity_data=[],
            phase_start_time=time.time(),
            phase_duration=self.config['calibration']['duration'] / 2
        )

        # 启动校准线程
        self.calibrating = True
        self.calibration_thread = threading.Thread(
            target=self._calibration_worker,
            daemon=True
        )
        self.calibration_thread.start()

        return True

    def _calibration_worker(self):
        """校准工作线程"""
        session = self.current_session
        if not session:
            return

        try:
            # 欢迎界面
            self.guide.display_message('welcome')
            input()  # 等待用户确认

            # 静息阶段
            self._run_rest_phase(session)

            # 活动阶段
            self._run_activity_phase(session)

            # 生成校准档案
            self._generate_calibration_profile(session)

            # 显示完成信息
            self._show_completion_message(session)

        except Exception as e:
            logger.error(f"校准过程失败: {e}")
            self.guide.display_message('error', error_message=str(e))

        finally:
            self.calibrating = False

    def _run_rest_phase(self, session: CalibrationSession):
        """运行静息校准阶段"""
        session.phase = 'rest'
        session.phase_start_time = time.time()

        self.guide.display_message('rest_intro')
        time.sleep(3)  # 给用户时间准备

        phase_duration = session.phase_duration
        start_time = time.time()

        while (time.time() - start_time) < phase_duration:
            # 模拟数据采集（实际应从传感器获取）
            emg_sample, gsr_sample = self._collect_sample()

            # 质量检查
            if self._validate_sample(emg_sample, gsr_sample):
                session.emg_rest_data.append(np.array(emg_sample))
                session.gsr_rest_data.append(gsr_sample)

            # 显示倒计时
            remaining = int(phase_duration - (time.time() - start_time))
            if remaining % 5 == 0:
                self.guide.display_message('countdown', seconds=remaining)

            time.sleep(0.1)

    def _run_activity_phase(self, session: CalibrationSession):
        """运行活动校准阶段"""
        session.phase = 'activity'
        session.phase_start_time = time.time()

        self.guide.display_message('activity_intro')
        time.sleep(3)  # 给用户时间准备

        phase_duration = session.phase_duration
        start_time = time.time()

        while (time.time() - start_time) < phase_duration:
            # 模拟数据采集
            emg_sample, gsr_sample = self._collect_sample()

            # 活动时EMG应更高，简单模拟
            emg_sample = [x * (1.5 + np.random.rand() * 0.5) for x in emg_sample]
            gsr_sample *= (1.1 + np.random.rand() * 0.2)

            # 质量检查
            if self._validate_sample(emg_sample, gsr_sample):
                session.emg_activity_data.append(np.array(emg_sample))
                session.gsr_activity_data.append(gsr_sample)

            # 显示倒计时
            remaining = int(phase_duration - (time.time() - start_time))
            if remaining % 5 == 0:
                self.guide.display_message('countdown', seconds=remaining)

            time.sleep(0.1)

    def _collect_sample(self) -> Tuple[List[float], float]:
        """采集单个样本（模拟）"""
        # 模拟EMG信号（8通道）
        emg_sample = [np.random.randn() * 0.3 for _ in range(8)]

        # 模拟GSR信号
        gsr_sample = 0.15 + np.random.randn() * 0.02

        return emg_sample, gsr_sample

    def _validate_sample(self, emg_sample: List[float], gsr_sample: float) -> bool:
        """验证样本质量"""
        # 基本质量检查
        if any(abs(x) > 5.0 for x in emg_sample):
            return False  # EMG异常值

        if gsr_sample <= 0 or gsr_sample > 2.0:
            return False  # GSR异常值

        return True

    def _generate_calibration_profile(self, session: CalibrationSession):
        """生成校准档案"""
        session.phase = 'processing'

        # 检查数据充足性
        if len(session.emg_rest_data) < 50 or len(session.emg_activity_data) < 50:
            raise ValueError("校准数据不足，请重新尝试")

        try:
            # 处理EMG数据
            emg_rest_array = np.array(session.emg_rest_data)
            emg_activity_array = np.array(session.emg_activity_data)

            # 计算各通道基准值
            emg_baseline_rms = {}
            emg_baseline_mdf = {}
            emg_quantiles = {}

            for ch in range(emg_rest_array.shape[1]):
                rest_channel = emg_rest_array[:, ch]
                activity_channel = emg_activity_array[:, ch]

                # RMS基准值
                emg_baseline_rms[f'channel_{ch}'] = float(np.mean(np.sqrt(np.mean(rest_channel ** 2))))

                # MDF基准值
                mdf_rest = self._calculate_channel_mdf(rest_channel)
                emg_baseline_mdf[f'channel_{ch}'] = mdf_rest

                # 归一化分位数
                all_channel_data = np.concatenate([rest_channel, activity_channel])
                p10, p90 = np.percentile(np.abs(all_channel_data), [10, 90])
                emg_quantiles[f'channel_{ch}'] = {'p10': float(p10), 'p90': float(p90)}

            # 处理GSR数据
            gsr_rest_array = np.array(session.gsr_rest_data)
            gsr_activity_array = np.array(session.gsr_activity_data)

            gsr_tonic_baseline = float(np.mean(gsr_rest_array))
            gsr_all_data = np.concatenate([gsr_rest_array, gsr_activity_array])
            gsr_p10, gsr_p90 = np.percentile(gsr_all_data, [10, 90])
            gsr_quantiles = {'p10': float(gsr_p10), 'p90': float(gsr_p90)}

            # 计算SCR阈值
            gsr_scr_threshold = float(np.std(gsr_activity_array) * 2)

            # 创建校准档案
            session.calibration_profile = CalibrationProfile(
                user_id=session.user_id,
                timestamp=time.time(),
                emg_baseline_rms=emg_baseline_rms,
                emg_baseline_mdf=emg_baseline_mdf,
                gsr_baseline_tonic=gsr_tonic_baseline,
                gsr_scr_threshold=gsr_scr_threshold,
                emg_quantiles=emg_quantiles,
                gsr_quantiles=gsr_quantiles,
                snr_threshold=self.config['quality']['min_snr'],
                clipping_threshold=self.config['quality']['max_clipping_rate']
            )

            # 保存校准档案
            profile_path = f"calibration_profile_{session.user_id}_{int(session.calibration_profile.timestamp)}.json"
            self._save_calibration_profile(session.calibration_profile, profile_path)

            session.phase = 'completed'

        except Exception as e:
            logger.error(f"生成校准档案失败: {e}")
            raise

    def _calculate_channel_mdf(self, channel_data: np.ndarray) -> float:
        """计算单通道中位频率"""
        try:
            from scipy.signal import welch
            f, P = welch(channel_data, fs=1000, nperseg=256, noverlap=128)
            c = np.cumsum(P) / np.sum(P)
            return float(f[np.searchsorted(c, 0.5)])
        except:
            return 100.0  # 默认值

    def _save_calibration_profile(self, profile: CalibrationProfile, filename: str):
        """保存校准档案"""
        profile_dict = asdict(profile)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(profile_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"校准档案已保存: {filename}")
        except Exception as e:
            logger.error(f"保存校准档案失败: {e}")

    def _show_completion_message(self, session: CalibrationSession):
        """显示校准完成信息"""
        if not session.calibration_profile:
            return

        # 计算稳定性评分
        rest_score = self._calculate_stability_score(session.emg_rest_data, session.gsr_rest_data)
        activity_score = self._calculate_stability_score(session.emg_activity_data, session.gsr_activity_data)
        overall_score = (rest_score + activity_score) / 2

        profile_path = f"calibration_profile_{session.user_id}_{int(session.calibration_profile.timestamp)}.json"

        self.guide.display_message(
            'completed',
            rest_score=rest_score,
            activity_score=activity_score,
            overall_score=overall_score,
            profile_path=profile_path
        )

    def _calculate_stability_score(self, emg_data: List[np.ndarray], gsr_data: List[float]) -> float:
        """计算数据稳定性评分"""
        if not emg_data or not gsr_data:
            return 0.0

        try:
            # EMG稳定性 (变异性)
            emg_array = np.array(emg_data)
            emg_cv = np.mean([np.std(channel) / max(np.mean(np.abs(channel)), 1e-6)
                             for channel in emg_array.T])

            # GSR稳定性
            gsr_cv = np.std(gsr_data) / max(np.mean(gsr_data), 1e-6)

            # 综合稳定性评分 (CV越小越稳定)
            stability = max(0, 100 - (emg_cv * 50 + gsr_cv * 30))
            return min(100, stability)

        except:
            return 50.0  # 默认中等稳定性

    def load_calibration_profile(self, profile_path: str) -> Optional[CalibrationProfile]:
        """加载校准档案"""
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_dict = json.load(f)

            profile = CalibrationProfile(**profile_dict)
            logger.info(f"已加载校准档案: {profile_path}")
            return profile

        except Exception as e:
            logger.error(f"加载校准档案失败: {e}")
            return None

    def get_available_profiles(self) -> List[str]:
        """获取可用的校准档案列表"""
        try:
            import glob
            profile_files = glob.glob("calibration_profile_*.json")
            return sorted(profile_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)
        except:
            return []

    def stop_calibration(self):
        """停止校准过程"""
        self.calibrating = False
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=1.0)


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    # 加载配置
    with open('signal_processing_config.json', 'r') as f:
        config = json.load(f)

    # 创建校准系统
    calibrator = CalibrationSystem(config)

    # 显示可用档案
    available_profiles = calibrator.get_available_profiles()
    if available_profiles:
        print("📁 可用的校准档案:")
        for profile in available_profiles[:5]:  # 显示最新5个
            print(f"  • {profile}")

    # 启动校准
    print("\n🎯 开始新的校准...")
    success = calibrator.start_calibration("test_user")

    if success:
        # 等待校准完成
        while calibrator.calibrating:
            time.sleep(0.5)

        print("✅ 校准过程完成")
    else:
        print("❌ 校准启动失败")