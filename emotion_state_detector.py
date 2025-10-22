#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 情绪状态检测器
Emotion State Detector

基于预处理特征进行实时情绪状态识别：
• 规则基线识别系统
• 平滑和鲁棒性处理
• 拒识和置信度评估
• 多模型集成接口

Author: EmotionHand Team
Version: 1.0.0
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionState(Enum):
    """情绪状态枚举"""
    RELAXED = "Relaxed"
    FOCUSED = "Focused"
    STRESSED = "Stressed"
    FATIGUED = "Fatigued"
    NEUTRAL = "Neutral"


@dataclass
class StatePrediction:
    """状态预测结果"""
    state: EmotionState
    confidence: float
    raw_scores: Dict[str, float]
    reasoning: str
    timestamp: float


class RuleBasedDetector:
    """基于规则的情绪状态检测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config['emotional_states']['thresholds']
        self.smoothing_config = config['emotional_states']['smoothing']

        # 平滑处理
        self.state_history = deque(maxlen=int(self.smoothing_config['voting_window_sec'] * 10))
        self.confidence_history = deque(maxlen=20)

        # 疲劳状态跟踪
        self.fatigue_start_time: Optional[float] = None
        self.rms_decline_rate = 0.0

        # 状态切换计数器
        self.transition_count = 0
        self.last_state = EmotionState.NEUTRAL

        logger.info("规则基检测器初始化完成")

    def predict_state(self, features: Dict[str, float], emg_features: Dict, gsr_features: Dict) -> StatePrediction:
        """基于规则预测情绪状态"""
        timestamp = time.time()

        # 计算原始分数
        raw_scores = self._calculate_raw_scores(features, emg_features, gsr_features)

        # 应用平滑处理
        smoothed_scores = self._apply_smoothing(raw_scores)

        # 疲劳特殊检测
        if self._detect_fatigue(features, smoothed_scores):
            final_state = EmotionState.FATIGUED
            confidence = max(0.7, smoothed_scores.get('fatigued', 0.5))
            reasoning = "肌肉活动下降 + 频率降低，持续时间≥30秒"

        else:
            # 选择最高分数的状态
            best_state_name = max(smoothed_scores, key=smoothed_scores.get)
            # 确保状态名匹配枚举
            state_mapping = {
                'relaxed': 'Relaxed',
                'focused': 'Focused',
                'stressed': 'Stressed',
                'fatigued': 'Fatigued',
                'neutral': 'Neutral'
            }
            final_state = EmotionState(state_mapping.get(best_state_name, 'Neutral'))
            confidence = smoothed_scores.get(best_state_name, 0.5)

            # 拒识处理
            if confidence < self.smoothing_config['rejection_threshold']:
                final_state = EmotionState.NEUTRAL
                confidence = 1.0 - confidence
                reasoning = f"低置信度，原最高分: {best_state_name} ({smoothed_scores[best_state_name]:.2f})"
            else:
                reasoning = self._generate_reasoning(final_state, smoothed_scores, features)

        # 更新历史记录
        self._update_history(final_state, confidence, timestamp)

        return StatePrediction(
            state=final_state,
            confidence=confidence,
            raw_scores=raw_scores,
            reasoning=reasoning,
            timestamp=timestamp
        )

    def _calculate_raw_scores(self, features: Dict[str, float], emg_features: Dict, gsr_features: Dict) -> Dict[str, float]:
        """计算各状态的原始分数"""
        scores = {}

        # 提取关键特征
        rms = features.get('rms', 0.5)
        mdf = features.get('mdf', 0.5)
        gsr_tonic = features.get('gsr_tonic', 0.5)

        # 放松状态
        relax_score = self._calculate_relax_score(rms, gsr_tonic, mdf)
        scores['relaxed'] = relax_score

        # 专注状态
        focus_score = self._calculate_focus_score(rms, gsr_tonic, mdf)
        scores['focused'] = focus_score

        # 紧张状态
        stress_score = self._calculate_stress_score(rms, gsr_tonic, mdf)
        scores['stressed'] = stress_score

        # 疲劳状态 (由特殊检测处理)
        scores['fatigued'] = 0.3  # 基础分数

        # 标准化分数
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total

        return scores

    def _calculate_relax_score(self, rms: float, gsr_tonic: float, mdf: float) -> float:
        """计算放松状态分数"""
        relax_threshold = self.thresholds['relaxed']
        score = 1.0

        # 低RMS有利于放松
        if rms > relax_threshold['rms_max']:
            penalty = (rms - relax_threshold['rms_max']) * 2
            score = max(0, score - penalty)
        else:
            reward = (relax_threshold['rms_max'] - rms) * 1.5
            score = min(1, score + reward)

        # 低GSR有利于放松
        if gsr_tonic > relax_threshold['gsr_max']:
            penalty = (gsr_tonic - relax_threshold['gsr_max']) * 2
            score = max(0, score - penalty)

        return score

    def _calculate_focus_score(self, rms: float, gsr_tonic: float, mdf: float) -> float:
        """计算专注状态分数"""
        focus_threshold = self.thresholds['focused']
        score = 0.5

        # RMS在中等范围内有利于专注
        rms_center = (focus_threshold['rms_min'] + focus_threshold['rms_max']) / 2
        rms_range = focus_threshold['rms_max'] - focus_threshold['rms_min']

        if focus_threshold['rms_min'] <= rms <= focus_threshold['rms_max']:
            reward = 1.0 - abs(rms - rms_center) / rms_range
            score = min(1, score + reward)
        else:
            penalty = abs(rms - rms_center) / rms_range
            score = max(0, score - penalty)

        # GSR在中等范围内有利于专注
        gsr_center = (focus_threshold['gsr_min'] + focus_threshold['gsr_max']) / 2
        gsr_range = focus_threshold['gsr_max'] - focus_threshold['gsr_min']

        if focus_threshold['gsr_min'] <= gsr_tonic <= focus_threshold['gsr_max']:
            reward = 1.0 - abs(gsr_tonic - gsr_center) / gsr_range
            score = min(1, score + reward * 0.5)

        # MDF较高有利于专注
        if mdf >= focus_threshold.get('mdf_min', 0.5):
            reward = (mdf - focus_threshold['mdf_min']) * 0.5
            score = min(1, score + reward)

        return score

    def _calculate_stress_score(self, rms: float, gsr_tonic: float, mdf: float) -> float:
        """计算紧张状态分数"""
        stress_threshold = self.thresholds['stressed']
        score = 0.3

        # 高RMS有利于紧张
        if rms >= stress_threshold['rms_min']:
            reward = (rms - stress_threshold['rms_min']) * 2
            score = min(1, score + reward)

        # 高GSR有利于紧张
        if gsr_tonic >= stress_threshold['gsr_min']:
            reward = (gsr_tonic - stress_threshold['gsr_min']) * 1.5
            score = min(1, score + reward)

        # 高MDF有利于紧张
        if mdf >= stress_threshold.get('mdf_min', 0.6):
            reward = (mdf - stress_threshold['mdf_min']) * 1.0
            score = min(1, score + reward)

        return score

    def _detect_fatigue(self, features: Dict[str, float], scores: Dict[str, float]) -> bool:
        """检测疲劳状态"""
        rms = features.get('rms', 0.5)
        mdf = features.get('mdf', 0.5)

        # 跟踪RMS下降趋势
        if len(self.confidence_history) >= 10:
            recent_confidences = list(self.confidence_history)[-10:]
            if len(recent_confidences) >= 5:
                # 计算趋势
                recent_rms = [features.get('rms', 0.5)] * 5  # 简化版本
                if len(recent_rms) >= 3:
                    self.rms_decline_rate = (recent_rms[-1] - recent_rms[-3]) / 3

        # 疲劳检测条件
        fatigue_conditions = [
            mdf < self.thresholds['fatigued']['mdf_max'],  # MDF降低
            rms < 0.3,  # RMS较低
            self.rms_decline_rate < -0.02  # RMS下降趋势
        ]

        if all(fatigue_conditions):
            if self.fatigue_start_time is None:
                self.fatigue_start_time = time.time()
            else:
                # 检查持续时间
                duration = time.time() - self.fatigue_start_time
                if duration >= self.thresholds['fatigued']['duration_min']:
                    return True
        else:
            self.fatigue_start_time = None

        return False

    def _apply_smoothing(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """应用平滑处理"""
        # 指数平滑
        alpha = self.smoothing_config['alpha']

        if len(self.state_history) > 0:
            last_scores = self.state_history[-1].raw_scores
            smoothed_scores = {}

            for state_name, score in raw_scores.items():
                last_score = last_scores.get(state_name, score)
                smoothed_score = alpha * score + (1 - alpha) * last_score
                smoothed_scores[state_name] = smoothed_score

            return smoothed_scores
        else:
            return raw_scores.copy()

    def _generate_reasoning(self, state: EmotionState, scores: Dict[str, float], features: Dict[str, float]) -> str:
        """生成状态推理说明"""
        rms = features.get('rms', 0.5)
        mdf = features.get('mdf', 0.5)
        gsr_tonic = features.get('gsr_tonic', 0.5)

        # 转换状态名用于查找分数
        state_mapping_rev = {
            'Relaxed': 'relaxed',
            'Focused': 'focused',
            'Stressed': 'stressed',
            'Fatigued': 'fatigued',
            'Neutral': 'neutral'
        }
        state_key = state_mapping_rev.get(state.value, 'neutral')
        confidence = float(scores.get(state_key, scores.get('neutral', 0.5)))

        reasoning_parts = []

        # 基础置信度
        reasoning_parts.append(f"置信度: {confidence:.2f}")

        # 具体特征说明
        if state == EmotionState.RELAXED:
            reasoning_parts.append(f"低肌肉活动 (RMS: {rms:.2f})")
            reasoning_parts.append(f"低皮电反应 (GSR: {gsr_tonic:.2f})")

        elif state == EmotionState.FOCUSED:
            reasoning_parts.append(f"中等肌肉活动 (RMS: {rms:.2f})")
            reasoning_parts.append(f"稳定皮电水平 (GSR: {gsr_tonic:.2f})")
            if mdf >= 0.5:
                reasoning_parts.append(f"良好肌肉激活频率 (MDF: {mdf:.2f})")

        elif state == EmotionState.STRESSED:
            reasoning_parts.append(f"高肌肉活动 (RMS: {rms:.2f})")
            reasoning_parts.append(f"高皮电反应 (GSR: {gsr_tonic:.2f})")
            if mdf >= 0.6:
                reasoning_parts.append(f"高频肌肉激活 (MDF: {mdf:.2f})")

        elif state == EmotionState.FATIGUED:
            reasoning_parts.append(f"肌肉活动下降 (RMS: {rms:.2f})")
            reasoning_parts.append(f"肌肉频率降低 (MDF: {mdf:.2f})")
            reasoning_parts.append("持续疲劳状态")

        return "; ".join(reasoning_parts)

    def _update_history(self, state: EmotionState, confidence: float, timestamp: float):
        """更新历史记录"""
        prediction = StatePrediction(
            state=state,
            confidence=confidence,
            raw_scores={},
            reasoning="",
            timestamp=timestamp
        )

        self.state_history.append(prediction)
        self.confidence_history.append(confidence)

        # 状态切换检测
        if state != self.last_state:
            self.transition_count += 1
            self.last_state = state

    def get_state_statistics(self) -> Dict:
        """获取状态统计信息"""
        if not self.state_history:
            return {}

        # 状态分布
        state_counts = {}
        for prediction in self.state_history:
            state_name = prediction.state.value
            state_counts[state_name] = state_counts.get(state_name, 0) + 1

        total_predictions = len(self.state_history)
        state_distribution = {
            state: count / total_predictions
            for state, count in state_counts.items()
        }

        # 平均置信度
        avg_confidence = np.mean(list(self.confidence_history)) if self.confidence_history else 0.0

        # 状态切换频率
        transition_rate = self.transition_count / max(1, total_predictions)

        return {
            'total_predictions': total_predictions,
            'state_distribution': state_distribution,
            'average_confidence': avg_confidence,
            'transition_rate': transition_rate,
            'most_common_state': max(state_distribution, key=state_distribution.get) if state_distribution else None
        }

    def reset_history(self):
        """重置历史记录"""
        self.state_history.clear()
        self.confidence_history.clear()
        self.transition_count = 0
        self.last_state = EmotionState.NEUTRAL
        self.fatigue_start_time = None
        self.rms_decline_rate = 0.0


class EnsembleDetector:
    """集成检测器（支持多模型集成）"""

    def __init__(self, config: Dict):
        self.config = config
        self.rule_based_detector = RuleBasedDetector(config)
        self.ml_detectors = {}  # 未来可添加ML模型

        # 集成权重
        self.ensemble_weights = {
            'rule_based': 0.8,
            'ml_models': 0.2
        }

        logger.info("集成检测器初始化完成")

    def predict_state(self, features: Dict[str, float], emg_features: Dict, gsr_features: Dict) -> StatePrediction:
        """集成预测情绪状态"""
        # 规则基线预测
        rule_prediction = self.rule_based_detector.predict_state(features, emg_features, gsr_features)

        # 目前只使用规则基线，未来可集成ML模型
        # TODO: 添加ML模型集成
        ml_scores = {}

        # 集成结果
        final_scores = self._ensemble_predictions(
            rule_prediction.raw_scores,
            ml_scores
        )

        # 选择最终状态
        best_state_name = max(final_scores, key=final_scores.get)
        # 确保状态名匹配枚举
        state_mapping = {
            'relaxed': 'Relaxed',
            'focused': 'Focused',
            'stressed': 'Stressed',
            'fatigued': 'Fatigued',
            'neutral': 'Neutral'
        }
        final_state = EmotionState(state_mapping.get(best_state_name, 'Neutral'))
        final_confidence = final_scores[best_state_name]

        return StatePrediction(
            state=final_state,
            confidence=final_confidence,
            raw_scores=final_scores,
            reasoning=f"集成预测: 规则基线({rule_prediction.confidence:.2f})",
            timestamp=time.time()
        )

    def _ensemble_predictions(self, rule_scores: Dict[str, float], ml_scores: Dict[str, float]) -> Dict[str, float]:
        """集成多个预测结果"""
        if not ml_scores:
            return rule_scores

        # 加权平均
        ensemble_scores = {}
        all_states = set(rule_scores.keys()) | set(ml_scores.keys())

        for state in all_states:
            rule_score = rule_scores.get(state, 0.0)
            ml_score = ml_scores.get(state, 0.0)

            ensemble_scores[state] = (
                self.ensemble_weights['rule_based'] * rule_score +
                self.ensemble_weights['ml_models'] * ml_score
            )

        # 标准化
        total = sum(ensemble_scores.values())
        if total > 0:
            for state in ensemble_scores:
                ensemble_scores[state] /= total

        return ensemble_scores


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)

    # 加载配置
    with open('signal_processing_config.json', 'r') as f:
        config = json.load(f)

    # 创建检测器
    detector = EnsembleDetector(config)

    print("🧪 测试情绪状态检测器...")

    # 模拟不同情绪的特征
    test_cases = {
        '放松': {'rms': 0.1, 'mdf': 0.3, 'gsr_tonic': 0.15},
        '专注': {'rms': 0.4, 'mdf': 0.7, 'gsr_tonic': 0.4},
        '紧张': {'rms': 0.8, 'mdf': 0.8, 'gsr_tonic': 0.7},
        '疲劳': {'rms': 0.2, 'mdf': 0.2, 'gsr_tonic': 0.3}
    }

    for case_name, features in test_cases.items():
        prediction = detector.predict_state(
            features,
            {}, {}  # 简化测试
        )

        print(f"\n📊 {case_name}状态测试:")
        print(f"  预测状态: {prediction.state.value}")
        print(f"  置信度: {prediction.confidence:.2f}")
        print(f"  推理: {prediction.reasoning}")

    # 显示统计
    stats = detector.rule_based_detector.get_state_statistics()
    print(f"\n📈 检测统计:")
    print(f"  总预测数: {stats['total_predictions']}")
    print(f"  平均置信度: {stats['average_confidence']:.2f}")
    print(f"  状态切换率: {stats['transition_rate']:.2f}")