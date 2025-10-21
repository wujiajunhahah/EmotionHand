#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个性化校准脚本
2分钟快速个体校准，解决电极位置差异
基于分位归一化和Few-shot微调
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import logging

# 导入自定义模块
from data_collection import DataCollector
from feature_extraction import UnifiedFeatureExtractor
from training import ModelTrainer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalCalibrator:
    """个性化校准器"""

    def __init__(self, base_model_path: str = None, config: Dict = None):
        # 默认配置
        self.config = {
            # 校准参数
            'calibration_duration_rest': 60,    # 静息状态时长(秒)
            'calibration_duration_gesture': 15, # 每个手势时长(秒)
            'calibration_duration_state': 15,   # 每个状态时长(秒)

            # 归一化参数
            'percentile_low': 10,     # 低分位数
            'percentile_high': 90,    # 高分位数

            # 微调参数
            'fine_tune_lr': 0.01,     # 微调学习率
            'fine_tune_epochs': 50,   # 微调轮数
            'freeze_backbone': True,  # 冻结骨干网络

            # 基础模型路径
            'base_model_dir': './models',
            'calibration_output_dir': './models/calibration',

            # 校准任务
            'calibration_gestures': ['Fist', 'Open'],      # 基础手势
            'calibration_states': ['Relaxed', 'Focused'],  # 基础状态
        }

        if config:
            self.config.update(config)

        # 创建输出目录
        os.makedirs(self.config['calibration_output_dir'], exist_ok=True)

        # 初始化组件
        self.feature_extractor = UnifiedFeatureExtractor()
        self.data_collector = DataCollector()

        # 加载基础模型
        self.base_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.load_base_models()

        # 校准参数
        self.normalization_params = {}
        self.calibrated_models = {}

        logger.info("个性化校准器初始化完成")

    def load_base_models(self):
        """加载基础模型"""
        try:
            # 加载手势分类器
            gesture_model_path = os.path.join(self.config['base_model_dir'], 'gesture_lightgbm.joblib')
            if os.path.exists(gesture_model_path):
                self.base_models['gesture'] = joblib.load(gesture_model_path)
                logger.info("基础手势模型加载成功")
            else:
                logger.warning(f"基础手势模型不存在: {gesture_model_path}")

            # 加载状态分类器
            state_model_path = os.path.join(self.config['base_model_dir'], 'state_lightgbm.joblib')
            if os.path.exists(state_model_path):
                self.base_models['state'] = joblib.load(state_model_path)
                logger.info("基础状态模型加载成功")
            else:
                logger.warning(f"基础状态模型不存在: {state_model_path}")

            # 加载标准化器
            scaler_path = os.path.join(self.config['base_model_dir'], 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scalers['base'] = joblib.load(scaler_path)
                logger.info("基础标准化器加载成功")
            else:
                self.scalers['base'] = StandardScaler()
                logger.info("使用新的标准化器")

            # 加载标签编码器
            encoder_path = os.path.join(self.config['base_model_dir'], 'label_encoder.joblib')
            if os.path.exists(encoder_path):
                self.label_encoders = joblib.load(encoder_path)
                logger.info("标签编码器加载成功")
            else:
                self.label_encoders = {'gesture': None, 'state': None}
                logger.info("使用新的标签编码器")

        except Exception as e:
            logger.error(f"基础模型加载失败: {e}")

    def collect_calibration_data(self, task_type: str) -> Dict:
        """采集校准数据"""
        print(f"\n=== {task_type.upper()} 校准数据采集 ===")

        # 启动数据采集
        self.data_collector.start()

        calibration_data = {
            'features': [],
            'labels': [],
            'timestamps': []
        }

        try:
            if task_type == 'normalization':
                # 分位归一化校准：静息 + 轻握
                print("第一步：静息状态校准 (60秒)")
                print("请保持手臂放松，不要有任何动作...")
                input("按回车开始静息状态采集")

                self.data_collector.start_collection('Calibration_Rest',
                                                   self.config['calibration_duration_rest'])
                time.sleep(self.config['calibration_duration_rest'])
                self.data_collector.stop_collection()

                print("\n第二步：轻握状态校准 (60秒)")
                print("请保持轻微握拳状态，不要太用力...")
                input("按回车开始轻握状态采集")

                self.data_collector.start_collection('Calibration_LightGrip',
                                                   self.config['calibration_duration_rest'])
                time.sleep(self.config['calibration_duration_rest'])
                self.data_collector.stop_collection()

            elif task_type == 'gesture':
                # 手势校准
                for gesture in self.config['calibration_gestures']:
                    print(f"\n采集手势: {gesture} ({self.config['calibration_duration_gesture']}秒)")
                    print(f"请执行 '{gesture}' 手势并保持...")
                    input("按回车开始采集")

                    self.data_collector.start_collection(f'Calibration_{gesture}',
                                                       self.config['calibration_duration_gesture'])
                    time.sleep(self.config['calibration_duration_gesture'])
                    self.data_collector.stop_collection()

            elif task_type == 'state':
                # 状态校准
                for state in self.config['calibration_states']:
                    print(f"\n采集状态: {state} ({self.config['calibration_duration_state']}秒)")
                    print(f"请进入 '{state}' 状态并保持...")
                    input("按回车开始采集")

                    self.data_collector.start_collection(f'Calibration_{state}',
                                                       self.config['calibration_duration_state'])
                    time.sleep(self.config['calibration_duration_state'])
                    self.data_collector.stop_collection()

        except KeyboardInterrupt:
            print("\n用户中断校准采集")
        except Exception as e:
            logger.error(f"校准数据采集错误: {e}")
        finally:
            self.data_collector.stop()

        # 加载采集的校准数据
        return self.load_calibration_data(task_type)

    def load_calibration_data(self, task_type: str) -> Dict:
        """加载校准数据"""
        calibration_data = {
            'features': [],
            'labels': [],
            'timestamps': []
        }

        data_dir = self.data_collector.config['output_dir']
        session_files = [f for f in os.listdir(data_dir)
                        if f.startswith(self.data_collector.config['session_name']) and f.endswith('.csv')]

        for file in session_files:
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    # 提取特征
                    emg_data = df[[f'emg{i}' for i in range(1, 9)]].values
                    gsr_data = df['gsr'].values

                    features, _, _ = self.feature_extractor.extract_combined_features(
                        emg_data, gsr_data
                    )

                    labels = df['label'].values
                    timestamps = df['timestamp'].values

                    # 对齐特征和标签数量
                    min_len = min(len(features), len(labels))
                    calibration_data['features'].extend(features[:min_len])
                    calibration_data['labels'].extend(labels[:min_len])
                    calibration_data['timestamps'].extend(timestamps[:min_len])

            except Exception as e:
                logger.error(f"加载校准数据文件错误 {file}: {e}")

        # 转换为numpy数组
        calibration_data['features'] = np.array(calibration_data['features'])
        calibration_data['labels'] = np.array(calibration_data['labels'])

        logger.info(f"加载校准数据: {len(calibration_data['features'])} 样本")
        return calibration_data

    def compute_normalization_params(self, data: Dict) -> Dict:
        """计算归一化参数"""
        print("\n=== 计算归一化参数 ===")

        # 分离静息和轻握数据
        rest_mask = data['labels'] == 'Calibration_Rest'
        grip_mask = data['labels'] == 'Calibration_LightGrip'

        if not np.any(rest_mask) or not np.any(grip_mask):
            logger.error("缺少静息或轻握数据，无法计算归一化参数")
            return {}

        rest_features = data['features'][rest_mask]
        grip_features = data['features'][grip_mask]

        # 计算每个特征的分位数
        rest_low = np.percentile(rest_features, self.config['percentile_low'], axis=0)
        rest_high = np.percentile(rest_features, self.config['percentile_high'], axis=0)

        grip_low = np.percentile(grip_features, self.config['percentile_low'], axis=0)
        grip_high = np.percentile(grip_features, self.config['percentile_high'], axis=0)

        # 合并计算整体归一化参数
        overall_low = np.minimum(rest_low, grip_low)
        overall_high = np.maximum(rest_high, grip_high)

        # 避免除零
        range_values = overall_high - overall_low
        range_values[range_values < 1e-6] = 1.0

        normalization_params = {
            'low_percentile': overall_low,
            'high_percentile': overall_high,
            'range': range_values,
            'method': 'percentile_normalization',
            'percentile_low': self.config['percentile_low'],
            'percentile_high': self.config['percentile_high']
        }

        self.normalization_params = normalization_params

        print(f"归一化参数计算完成:")
        print(f"  特征维度: {len(overall_low)}")
        print(f"  低分位数: {overall_low[:5]}...")  # 显示前5个
        print(f"  高分位数: {overall_high[:5]}...")

        return normalization_params

    def apply_normalization(self, features: np.ndarray) -> np.ndarray:
        """应用归一化"""
        if not self.normalization_params:
            logger.warning("归一化参数未计算，返回原始特征")
            return features

        normalized = (features - self.normalization_params['low_percentile']) / \
                    self.normalization_params['range']

        # 限制在合理范围内
        normalized = np.clip(normalized, -2.0, 2.0)

        return normalized

    def fine_tune_model(self, data: Dict, model_type: str) -> bool:
        """微调模型"""
        print(f"\n=== 微调 {model_type} 模型 ===")

        if model_type not in self.base_models:
            logger.error(f"缺少基础 {model_type} 模型")
            return False

        # 准备数据
        features = data['features']
        labels = data['labels']

        # 过滤相关标签
        if model_type == 'gesture':
            valid_labels = self.config['calibration_gestures']
            prefix = 'Calibration_'
        else:  # state
            valid_labels = self.config['calibration_states']
            prefix = 'Calibration_'

        mask = np.array([label.replace(prefix, '') in valid_labels for label in labels])
        if not np.any(mask):
            logger.error(f"没有找到有效的 {model_type} 校准数据")
            return False

        X = features[mask]
        y = np.array([label.replace(prefix, '') for label in labels[mask]])

        # 应用归一化
        X_normalized = self.apply_normalization(X)

        print(f"微调数据: {X_normalized.shape[0]} 样本, {X_normalized.shape[1]} 特征")
        print(f"类别分布: {np.unique(y, return_counts=True)}")

        try:
            # 获取基础模型
            base_model = self.base_models[model_type]

            # 创建新的微调模型
            fine_tuned_model = lgb.LGBMClassifier(
                objective='multiclass',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=self.config['fine_tune_lr'],
                n_estimators=100,  # 较少的迭代次数
                max_depth=-1,
                random_state=42,
                n_jobs=-1
            )

            # 划分训练/验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_normalized, y, test_size=0.3, random_state=42, stratify=y
            )

            # 训练微调模型
            fine_tuned_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            # 评估性能
            train_score = fine_tuned_model.score(X_train, y_train)
            val_score = fine_tuned_model.score(X_val, y_val)

            print(f"微调完成 - 训练准确率: {train_score:.3f}, 验证准确率: {val_score:.3f}")

            # 保存微调模型
            model_path = os.path.join(
                self.config['calibration_output_dir'],
                f'calibrated_{model_type}_model.joblib'
            )
            joblib.dump(fine_tuned_model, model_path)

            self.calibrated_models[model_type] = fine_tuned_model

            print(f"微调模型已保存: {model_path}")
            return True

        except Exception as e:
            logger.error(f"模型微调失败: {e}")
            return False

    def visualize_calibration_results(self, data: Dict):
        """可视化校准结果"""
        print("\n=== 生成校准结果可视化 ===")

        try:
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('个性化校准结果', fontsize=16)

            # 1. 归一化效果
            if len(data['features']) > 0:
                ax = axes[0, 0]
                sample_features = data['features'][:1000]  # 取前1000个样本

                if self.normalization_params:
                    normalized_features = self.apply_normalization(sample_features)
                    ax.hist(normalized_features[:, 0], bins=50, alpha=0.7, label='归一化后')
                    ax.hist(sample_features[:, 0], bins=50, alpha=0.7, label='原始')
                    ax.set_title('特征归一化效果')
                    ax.legend()
                else:
                    ax.hist(sample_features[:, 0], bins=50)
                    ax.set_title('原始特征分布')

            # 2. 标签分布
            ax = axes[0, 1]
            unique_labels, counts = np.unique(data['labels'], return_counts=True)
            ax.bar(unique_labels, counts)
            ax.set_title('校准数据标签分布')
            ax.set_xlabel('标签')
            ax.set_ylabel('样本数')
            plt.setp(ax.get_xticklabels(), rotation=45)

            # 3. 特征相关性
            if len(data['features']) > 0:
                ax = axes[1, 0]
                sample_features = data['features'][:500]  # 取前500个样本
                correlation_matrix = np.corrcoef(sample_features.T)
                im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                ax.set_title('特征相关性矩阵')
                plt.colorbar(im, ax=ax)

            # 4. 校准流程时间线
            ax = axes[1, 1]
            timestamps = data['timestamps']
            if len(timestamps) > 0:
                start_time = timestamps[0]
                relative_times = [(t - start_time) / 60 for t in timestamps]  # 转换为分钟
                ax.plot(relative_times, range(len(relative_times)), '.', markersize=1)
                ax.set_title('校准数据时间线')
                ax.set_xlabel('时间 (分钟)')
                ax.set_ylabel('样本序号')

            plt.tight_layout()

            # 保存图表
            plot_path = os.path.join(
                self.config['calibration_output_dir'],
                f'calibration_results_{int(time.time())}.png'
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"校准结果图表已保存: {plot_path}")

            plt.show()

        except Exception as e:
            logger.error(f"可视化生成失败: {e}")

    def run_calibration(self):
        """运行完整校准流程"""
        print("=== 开始个性化校准流程 ===")
        print("总校准时间: 约2分钟")
        print("校准步骤:")
        print("1. 静息状态校准 (60秒)")
        print("2. 轻握状态校准 (60秒)")
        print("3. 手势微调校准 (30秒)")
        print("4. 状态微调校准 (30秒)\n")

        input("准备好了吗？按回车开始校准...")

        start_time = time.time()

        try:
            # 步骤1: 归一化校准
            print("\n📊 步骤1: 计算归一化参数")
            normalization_data = self.collect_calibration_data('normalization')
            self.compute_normalization_params(normalization_data)

            # 步骤2: 手势微调
            print("\n✋ 步骤2: 手势模型微调")
            gesture_data = self.collect_calibration_data('gesture')
            if len(gesture_data['features']) > 0:
                self.fine_tune_model(gesture_data, 'gesture')

            # 步骤3: 状态微调
            print("\n😌 步骤3: 状态模型微调")
            state_data = self.collect_calibration_data('state')
            if len(state_data['features']) > 0:
                self.fine_tune_model(state_data, 'state')

            # 步骤4: 可视化结果
            print("\n📈 步骤4: 生成校准报告")
            all_data = {
                'features': np.concatenate([
                    normalization_data['features'],
                    gesture_data['features'],
                    state_data['features']
                ]) if len(gesture_data['features']) > 0 and len(state_data['features']) > 0 else normalization_data['features'],
                'labels': np.concatenate([
                    normalization_data['labels'],
                    gesture_data['labels'],
                    state_data['labels']
                ]) if len(gesture_data['features']) > 0 and len(state_data['features']) > 0 else normalization_data['labels'],
                'timestamps': np.concatenate([
                    normalization_data['timestamps'],
                    gesture_data['timestamps'],
                    state_data['timestamps']
                ]) if len(gesture_data['features']) > 0 and len(state_data['features']) > 0 else normalization_data['timestamps']
            }

            self.visualize_calibration_results(all_data)

            # 保存校准配置
            calibration_config = {
                'normalization_params': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                       for k, v in self.normalization_params.items()},
                'calibrated_models': list(self.calibrated_models.keys()),
                'calibration_time': time.time() - start_time,
                'session_name': self.data_collector.config['session_name'],
                'config': self.config
            }

            config_path = os.path.join(
                self.config['calibration_output_dir'],
                'calibration_config.json'
            )
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_config, f, indent=2, ensure_ascii=False)

            total_time = time.time() - start_time
            print(f"\n✅ 校准完成！")
            print(f"总用时: {total_time:.1f} 秒")
            print(f"校准配置已保存: {config_path}")
            print(f"微调模型数量: {len(self.calibrated_models)}")

        except Exception as e:
            logger.error(f"校准流程失败: {e}")
        finally:
            self.data_collector.stop()

    def load_calibration(self, config_path: str) -> bool:
        """加载已有校准配置"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.normalization_params = config['normalization_params']
            # 转换回numpy数组
            for key in ['low_percentile', 'high_percentile', 'range']:
                if key in self.normalization_params:
                    self.normalization_params[key] = np.array(self.normalization_params[key])

            # 加载微调模型
            for model_name in config['calibrated_models']:
                model_path = os.path.join(
                    self.config['calibration_output_dir'],
                    f'calibrated_{model_name}_model.joblib'
                )
                if os.path.exists(model_path):
                    self.calibrated_models[model_name] = joblib.load(model_path)

            print(f"校准配置加载成功: {config_path}")
            return True

        except Exception as e:
            logger.error(f"校准配置加载失败: {e}")
            return False

def main():
    """主函数"""
    print("=== EMG + GSR 个性化校准工具 ===\n")

    calibrator = PersonalCalibrator()

    while True:
        print("\n=== 校准菜单 ===")
        print("1. 运行完整校准流程 (推荐)")
        print("2. 仅重新采集校准数据")
        print("3. 加载已有校准配置")
        print("4. 查看校准状态")
        print("5. 退出")

        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == '1':
            calibrator.run_calibration()
        elif choice == '2':
            task_type = input("请选择校准类型 (normalization/gesture/state): ").strip()
            if task_type in ['normalization', 'gesture', 'state']:
                data = calibrator.collect_calibration_data(task_type)
                if task_type == 'normalization':
                    calibrator.compute_normalization_params(data)
                else:
                    calibrator.fine_tune_model(data, task_type)
            else:
                print("无效的校准类型")
        elif choice == '3':
            config_path = input("请输入校准配置文件路径: ").strip()
            if os.path.exists(config_path):
                calibrator.load_calibration(config_path)
            else:
                print("文件不存在")
        elif choice == '4':
            print(f"\n=== 校准状态 ===")
            print(f"归一化参数: {'已计算' if calibrator.normalization_params else '未计算'}")
            print(f"微调模型: {list(calibrator.calibrated_models.keys())}")
        elif choice == '5':
            break
        else:
            print("无效选择")

if __name__ == "__main__":
    main()