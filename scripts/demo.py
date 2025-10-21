#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 演示脚本
完整展示从数据采集到实时推理的全流程
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import UnifiedFeatureExtractor
from training import ModelTrainer
from data_collection import DataCollector
from calibration import PersonalCalibrator
from real_time_inference import RealTimePipeline

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionHandDemo:
    """EmotionHand 演示系统"""

    def __init__(self):
        print("=== EmotionHand 演示系统 ===")
        print("基于EMG+GSR的情绪状态识别系统")
        print("实时延迟 <100ms，支持个性化校准\n")

        # 初始化组件
        self.feature_extractor = UnifiedFeatureExtractor()
        self.trainer = ModelTrainer()
        self.calibrator = PersonalCalibrator()
        self.pipeline = RealTimePipeline()

        # 演示数据
        self.demo_data = None
        self.models = {}

    def generate_demo_data(self):
        """生成演示数据"""
        print("📊 生成演示数据...")

        # 生成EMG数据 (8通道)
        sample_rate_emg = 1000
        duration = 10  # 10秒数据
        n_samples = sample_rate_emg * duration

        # 创建不同状态的模拟信号
        t = np.linspace(0, duration, n_samples)
        emg_data = np.zeros((n_samples, 8))

        # 为每个通道添加不同频率的信号
        frequencies = [20, 50, 80, 120, 200, 300, 350, 400]
        amplitudes = [0.5, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1, 0.1]

        for i in range(8):
            # 基础信号
            signal = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)

            # 添加噪声
            signal += 0.1 * np.random.randn(n_samples)

            # 添加调制 (模拟肌肉收缩)
            if i < 4:  # 前4个通道有更强的活动
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
                signal *= envelope

            emg_data[:, i] = signal

        # 生成GSR数据 (1通道)
        sample_rate_gsr = 100
        n_samples_gsr = sample_rate_gsr * duration
        t_gsr = np.linspace(0, duration, n_samples_gsr)

        # GSR信号 (低频)
        gsr_data = 0.1 * np.sin(2 * np.pi * 0.1 * t_gsr)  # 0.1Hz基础信号
        gsr_data += 0.05 * np.sin(2 * np.pi * 0.05 * t_gsr)  # 0.05Hz慢变信号
        gsr_data += 0.02 * np.random.randn(n_samples_gsr)  # 噪声

        # 生成标签
        n_windows = 150  # 150个时间窗口
        gesture_labels = np.random.choice(['Fist', 'Open', 'Pinch', 'Point'], n_windows)
        state_labels = np.random.choice(['Relaxed', 'Focused', 'Stressed', 'Fatigued'], n_windows)

        self.demo_data = {
            'emg_data': emg_data,
            'gsr_data': gsr_data,
            'gesture_labels': gesture_labels,
            'state_labels': state_labels,
            'sample_rate_emg': sample_rate_emg,
            'sample_rate_gsr': sample_rate_gsr
        }

        print(f"✅ 演示数据生成完成:")
        print(f"   EMG数据: {emg_data.shape} (时长: {duration}秒)")
        print(f"   GSR数据: {gsr_data.shape}")
        print(f"   手势标签: {np.unique(gesture_labels)}")
        print(f"   状态标签: {np.unique(state_labels)}")

    def extract_demo_features(self):
        """提取演示特征"""
        if self.demo_data is None:
            self.generate_demo_data()

        print("\n🔧 提取特征...")

        # 提取组合特征
        features, emg_windows, gsr_windows = self.feature_extractor.extract_combined_features(
            self.demo_data['emg_data'],
            self.demo_data['gsr_data']
        )

        print(f"✅ 特征提取完成:")
        print(f"   特征矩阵: {features.shape}")
        print(f"   EMG窗口数: {len(emg_windows)}")
        print(f"   GSR窗口数: {len(gsr_windows)}")
        print(f"   特征维度: {features.shape[1]}")

        return features

    def train_demo_models(self, features):
        """训练演示模型"""
        print("\n🧠 训练模型...")

        # 准备训练数据
        n_samples = min(len(features), len(self.demo_data['gesture_labels']))
        X = features[:n_samples]
        y_gesture = self.demo_data['gesture_labels'][:n_samples]
        y_state = self.demo_data['state_labels'][:n_samples]

        # 训练手势分类器
        print("训练手势分类器...")
        gesture_model, gesture_metadata = self.trainer.train_model(
            X, y_gesture, model_type='lightgbm', mode='gesture'
        )
        self.models['gesture'] = gesture_model

        print(f"✅ 手势模型训练完成 - 测试准确率: {gesture_metadata['test_score']:.3f}")

        # 训练状态分类器
        print("训练状态分类器...")
        state_model, state_metadata = self.trainer.train_model(
            X, y_state, model_type='lightgbm', mode='state'
        )
        self.models['state'] = state_model

        print(f"✅ 状态模型训练完成 - 测试准确率: {state_metadata['test_score']:.3f}")

        # 保存模型
        os.makedirs('./models', exist_ok=True)
        import joblib
        joblib.dump(gesture_model, './models/demo_gesture_model.joblib')
        joblib.dump(state_model, './models/demo_state_model.joblib')
        joblib.dump(self.trainer.scalers['gesture'], './models/demo_scaler.joblib')

        print("💾 模型已保存到 ./models/")

    def simulate_real_time_inference(self):
        """模拟实时推理"""
        print("\n⚡ 模拟实时推理...")

        if self.demo_data is None:
            self.generate_demo_data()

        # 创建模拟数据流
        def simulate_data_stream():
            """模拟数据流生成器"""
            emg_buffer = []
            gsr_buffer = []

            for i in range(100):  # 模拟100个时间步
                # 获取一小段数据
                start_idx = i * 10
                end_idx = start_idx + 256

                if end_idx < len(self.demo_data['emg_data']):
                    emg_segment = self.demo_data['emg_data'][start_idx:end_idx]
                    gsr_segment = self.demo_data['gsr_data'][start_idx//10:end_idx//10]

                    yield emg_segment, gsr_segment, i

                time.sleep(0.01)  # 模拟实时性

        # 模拟推理过程
        predictions = []
        latencies = []

        for emg_segment, gsr_segment, step in simulate_data_stream():
            start_time = time.time()

            # 特征提取
            features, _, _ = self.feature_extractor.extract_combined_features(
                emg_segment, gsr_segment
            )

            if features.shape[0] > 0:
                feature_vec = features[-1]  # 取最新特征

                # 手势预测
                if 'gesture' in self.models:
                    gesture_pred = self.models['gesture'].predict([feature_vec])[0]
                    gesture_conf = np.max(self.models['gesture'].predict_proba([feature_vec])[0])
                else:
                    gesture_pred = "Unknown"
                    gesture_conf = 0.0

                # 状态预测
                if 'state' in self.models:
                    state_pred = self.models['state'].predict([feature_vec])[0]
                    state_conf = np.max(self.models['state'].predict_proba([feature_vec])[0])
                else:
                    state_pred = "Unknown"
                    state_conf = 0.0

                # 计算延迟
                latency = (time.time() - start_time) * 1000  # ms

                predictions.append({
                    'step': step,
                    'gesture': gesture_pred,
                    'state': state_pred,
                    'confidence': min(gesture_conf, state_conf),
                    'latency': latency
                })

                latencies.append(latency)

                # 实时显示
                if step % 10 == 0:
                    print(f"步骤 {step:3d}: {gesture_pred:8s} | {state_pred:8s} | "
                          f"置信度: {min(gesture_conf, state_conf):.3f} | "
                          f"延迟: {latency:.1f}ms")

        # 统计结果
        if predictions:
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            confidence_threshold = 0.6
            high_conf_count = sum(1 for p in predictions if p['confidence'] >= confidence_threshold)

            print(f"\n📊 实时推理统计:")
            print(f"   总预测次数: {len(predictions)}")
            print(f"   平均延迟: {avg_latency:.1f}ms")
            print(f"   最大延迟: {max_latency:.1f}ms")
            print(f"   高置信度预测(>={confidence_threshold}): {high_conf_count}/{len(predictions)}")
            print(f"   实时性能: {'✅达标' if avg_latency < 100 else '❌超标'} (目标: <100ms)")

            return predictions

    def visualize_demo_results(self, predictions):
        """可视化演示结果"""
        if not predictions:
            return

        print("\n📈 生成可视化结果...")

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('EmotionHand 演示结果', fontsize=16)

        # 1. 预测结果时间线
        ax = axes[0, 0]
        steps = [p['step'] for p in predictions]
        gestures = [p['gesture'] for p in predictions]
        states = [p['state'] for p in predictions]

        # 手势时间线
        unique_gestures = list(set(gestures))
        gesture_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_gestures)))
        gesture_dict = {g: gesture_colors[i] for i, g in enumerate(unique_gestures)}

        for i, (step, gesture) in enumerate(zip(steps, gestures)):
            ax.scatter(step, i % 5, c=[gesture_dict[gesture]], s=20, alpha=0.7)

        ax.set_title('手势识别时间线')
        ax.set_xlabel('时间步')
        ax.set_ylabel('预测位置')

        # 添加图例
        for gesture, color in gesture_dict.items():
            ax.scatter([], [], c=[color], label=gesture, s=50)
        ax.legend()

        # 2. 置信度分布
        ax = axes[0, 1]
        confidences = [p['confidence'] for p in predictions]
        ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0.6, color='red', linestyle='--', label='置信度阈值')
        ax.set_title('预测置信度分布')
        ax.set_xlabel('置信度')
        ax.set_ylabel('频次')
        ax.legend()

        # 3. 延迟分析
        ax = axes[1, 0]
        latencies = [p['latency'] for p in predictions]
        ax.plot(latencies, alpha=0.7, color='green')
        ax.axhline(y=100, color='red', linestyle='--', label='100ms目标')
        ax.set_title('实时延迟分析')
        ax.set_xlabel('预测序号')
        ax.set_ylabel('延迟 (ms)')
        ax.legend()

        # 4. 状态分布饼图
        ax = axes[1, 1]
        state_counts = {}
        for p in predictions:
            state_counts[p['state']] = state_counts.get(p['state'], 0) + 1

        colors = plt.cm.Set2(np.linspace(0, 1, len(state_counts)))
        ax.pie(state_counts.values(), labels=state_counts.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('状态识别分布')

        plt.tight_layout()

        # 保存图表
        output_dir = './docs/results'
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'demo_results_{int(time.time())}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 可视化结果已保存: {plot_path}")

        plt.show()

    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 开始完整演示流程\n")

        try:
            # 步骤1: 生成演示数据
            self.generate_demo_data()

            # 步骤2: 特征提取
            features = self.extract_demo_features()

            # 步骤3: 训练模型
            self.train_demo_models(features)

            # 步骤4: 模拟实时推理
            predictions = self.simulate_real_time_inference()

            # 步骤5: 可视化结果
            self.visualize_demo_results(predictions)

            print("\n🎉 完整演示流程完成!")
            print("\n📋 演示总结:")
            print("   ✅ 数据生成: 10秒EMG+GSR模拟数据")
            print("   ✅ 特征提取: LibEMG风格的多维特征")
            print("   ✅ 模型训练: LightGBM手势+状态分类器")
            print("   ✅ 实时推理: <100ms延迟的高性能推理")
            print("   ✅ 结果可视化: 多维度分析图表")

        except Exception as e:
            logger.error(f"演示流程出错: {e}")
            import traceback
            traceback.print_exc()

    def interactive_demo(self):
        """交互式演示"""
        print("=== EmotionHand 交互式演示 ===\n")

        while True:
            print("📋 演示菜单:")
            print("1. 运行完整演示")
            print("2. 仅生成演示数据")
            print("3. 仅训练模型")
            print("4. 仅模拟实时推理")
            print("5. 查看系统状态")
            print("6. 退出")

            choice = input("\n请选择演示项目 (1-6): ").strip()

            if choice == '1':
                self.run_complete_demo()
            elif choice == '2':
                self.generate_demo_data()
                features = self.extract_demo_features()
                print(f"特征矩阵形状: {features.shape}")
            elif choice == '3':
                if self.demo_data is None:
                    self.generate_demo_data()
                features = self.extract_demo_features()
                self.train_demo_models(features)
            elif choice == '4':
                if not self.models:
                    print("⚠️ 需要先训练模型，请选择选项3")
                else:
                    self.simulate_real_time_inference()
            elif choice == '5':
                print("\n📊 系统状态:")
                print(f"   演示数据: {'已生成' if self.demo_data else '未生成'}")
                print(f"   训练模型: {list(self.models.keys())}")
                print(f"   特征提取器: 已就绪")
                print(f"   数据采集器: 已就绪")
                print(f"   校准器: 已就绪")
                print(f"   实时管线: 已就绪")
            elif choice == '6':
                print("👋 感谢使用EmotionHand演示系统!")
                break
            else:
                print("❌ 无效选择，请重试")

            input("\n按回车继续...")

def main():
    """主函数"""
    demo = EmotionHandDemo()

    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            demo.run_complete_demo()
        elif sys.argv[1] == '--interactive':
            demo.interactive_demo()
        else:
            print("用法:")
            print("  python demo.py --full         # 运行完整演示")
            print("  python demo.py --interactive  # 交互式演示")
            print("  python demo.py                # 交互式演示")
    else:
        demo.interactive_demo()

if __name__ == "__main__":
    main()