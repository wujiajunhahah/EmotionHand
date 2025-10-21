#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG + GSR 模型训练脚本
支持多种算法和训练策略
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import lightgbm as lgb
import logging
from typing import Dict, Tuple, Optional, Any

# 导入自定义模块
from feature_extraction import extract_features_from_file, UnifiedFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.training_history = {}

    def prepare_data(self, data_files: List[str], mode: str = 'both') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """准备训练数据"""
        all_features = []
        all_labels = []
        label_mapping = {'gesture': [], 'state': []}

        logger.info(f"Loading {len(data_files)} data files for {mode} training")

        for file_path in data_files:
            try:
                features, labels = extract_features_from_file(file_path, UnifiedFeatureExtractor())
                all_features.append(features)

                if mode == 'gesture' or mode == 'both':
                    if labels['gesture'] is not None:
                        label_mapping['gesture'].extend(labels['gesture'])
                if mode == 'state' or mode == 'both':
                    if labels['state'] is not None:
                        label_mapping['state'].extend(labels['state'])

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        X = np.vstack(all_features)

        if mode == 'both':
            # 多任务学习：组合手势和状态标签
            y_gesture = np.array(label_mapping['gesture'])
            y_state = np.array(label_mapping['state'])
            y = {
                'gesture': y_gesture,
                'state': y_state
            }
        else:
            # 单任务学习
            if mode == 'gesture':
                y = np.array(label_mapping['gesture'])
            else:
                y = np.array(label_mapping['state'])

        return X, y, label_mapping

    def encode_labels(self, y: Any, label_type: str) -> Tuple[np.ndarray, Any]:
        """编码标签"""
        if isinstance(y, dict):
            # 多任务情况
            encoded = {}
            encoders = {}
            for key, values in y.items():
                encoder = LabelEncoder()
                encoded[key] = encoder.fit_transform(values)
                encoders[key] = encoder
            return encoded, encoders
        else:
            # 单任务情况
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(y)
            return encoded, {label_type: encoder}

    def train_lightgbm(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """训练LightGBM模型"""
        logger.info("Training LightGBM model")

        # 默认参数
        params = {
            'objective': 'multiclass',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)

        model = lgb.LGBMClassifier(**params)

        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")

        # 训练最终模型
        model.fit(X, y)

        return model, cv_scores

    def train_svm(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """训练SVM模型"""
        from sklearn.svm import SVC

        logger.info("Training SVM model")

        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        params.update(kwargs)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**params))
        ])

        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")

        # 训练最终模型
        pipeline.fit(X, y)

        return pipeline, cv_scores

    def train_lda(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """训练LDA模型"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        logger.info("Training LDA model")

        params = {
            'solver': 'svd',
            'shrinkage': 'auto',
            'covariance_estimator': 'ledoit_wolf'
        }
        params.update(kwargs)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(**params))
        ])

        # 交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f}")

        # 训练最终模型
        pipeline.fit(X, y)

        return pipeline, cv_scores

    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'lightgbm',
                    mode: str = 'gesture', **model_params) -> Tuple[Any, Dict]:
        """训练模型"""

        logger.info(f"Training {model_type} model for {mode} classification")

        # 编码标签
        y_encoded, label_encoder = self.encode_labels(y, mode)
        self.label_encoders[mode] = label_encoder

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # 训练模型
        if model_type == 'lightgbm':
            model, cv_scores = self.train_lightgbm(X_train, y_train, **model_params)
        elif model_type == 'svm':
            model, cv_scores = self.train_svm(X_train, y_train, **model_params)
        elif model_type == 'lda':
            model, cv_scores = self.train_lda(X_train, y_train, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 评估测试集
        test_score = model.score(X_test, y_test)
        logger.info(f"Test set score: {test_score:.4f}")

        # 保存模型和元数据
        self.models[mode] = model
        self.training_history[mode] = {
            'model_type': model_type,
            'cv_scores': cv_scores,
            'test_score': test_score,
            'feature_importance': None
        }

        # 获取特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            if hasattr(model, 'named_steps'):
                # 对于Pipeline，获取最后一步的特征重要性
                final_step = model.named_steps[-1][1]
                if hasattr(final_step, 'feature_importances_'):
                    importance = final_step.feature_importances_
                elif hasattr(final_step, 'coef_'):
                    importance = np.abs(final_step.coef_)
                else:
                    importance = None
            else:
                importance = None

            self.training_history[mode]['feature_importance'] = importance

        return model, {
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'y_train_shape': y_train.shape,
            'y_test_shape': y_test.shape,
            'label_encoder': label_encoder,
            'test_score': test_score,
            'cv_scores': cv_scores
        }

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, model: Any,
                        mode: str = 'gesture', label_encoder: Any = None) -> Dict:
        """评估模型性能"""

        if label_encoder is None:
            label_encoder = self.label_encoders.get(mode)

        # 预测
        y_pred = model.predict(X_test)

        # 解码标签
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        # 分类报告
        report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)

        # 混淆矩阵
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_test_decoded,
            'y_pred': y_pred_decoded
        }

    def plot_training_results(self, mode: str = 'gesture'):
        """绘制训练结果"""
        if mode not in self.training_history:
            logger.warning(f"No training history for mode: {mode}")
            return

        history = self.training_history[mode]
        cv_scores = history['cv_scores']

        plt.figure(figsize=(12, 4))

        # 交叉验证分数分布
        plt.subplot(1, 3, 1)
        sns.hist(cv_scores, alpha=0.7, bins=20)
        plt.axvline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
        plt.title(f'{mode} Cross-Validation F1 Scores Distribution')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.legend()

        # 训练历史
        if 'feature_importance' in history and history['feature_importance'] is not None:
            importance = history['feature_importance']
            plt.subplot(1, 3, 2)

            # 取前20个最重要特征
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            top_features = np.argsort(importance)[-20:]

            plt.barh(range(20), importance[top_features])
            plt.yticks(range(20), [feature_names[i] for i in top_features])
            plt.title(f'{mode} Top 20 Feature Importance')
            plt.xlabel('Importance')
        else:
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, 'No feature importance data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{mode} Feature Importance (Not Available)')

        # 测试分数对比
        plt.subplot(1, 3, 3)
        metrics = ['CV Mean', 'Test Score', 'Best CV']
        scores = [cv_scores.mean(), history['test_score'], cv_scores.max()]
        colors = ['blue', 'green', 'red']

        bars = plt.bar(metrics, scores, color=colors, alpha=0.7)
        plt.title(f'{mode} Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # 在柱状图上添加数值
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    # 配置参数
    config = {
        'data_dir': './data/public',
        'model_type': 'lightgbm',  # 'lightgbm', 'svm', 'lda'
        'mode': 'gesture',         # 'gesture', 'state', 'both'
        'model_output_dir': './models',
        'results_dir': './docs/results',
        'window_size_emg': 256,
        'step_size_emg': 64,
        'window_size_gsr': 25,
        'step_size_gsr': 5
    }

    # 创建输出目录
    os.makedirs(config['model_output_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    # 初始化训练器
    trainer = ModelTrainer()

    # 查找数据文件
    data_files = []
    for root, dirs, files in os.walk(config['data_dir']):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))

    if not data_files:
        logger.error("No data files found!")
        return

    logger.info(f"Found {len(data_files)} data files")

    # 准备数据
    X, y, label_mapping = trainer.prepare_data(data_files, config['mode'])

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Labels: {config['mode']} - {len(np.unique(y))} classes")

    # 训练模型
    model, metadata = trainer.train_model(X, y, config['model_type'], config['mode'])

    # 保存模型
    model_path = os.path.join(config['model_output_dir'], f"{config['mode']}_{config['model_type']}.joblib")
    joblib.dump(model, model_path)
    joblib.dump(metadata, model_path.replace('.joblib', '_metadata.joblib'))

    logger.info(f"Model saved to: {model_path}")

    # 评估模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    evaluation_results = trainer.evaluate_model(X_test, y_test, model, config['mode'])

    # 保存评估结果
    results_path = os.path.join(config['results_dir'], f"{config['mode']}_{config['model_type']}_results.pkl")
    joblib.dump(evaluation_results, results_path)

    # 绘制训练结果
    trainer.plot_training_results(config['mode'])

    logger.info("Training completed!")

if __name__ == "__main__":
    main()