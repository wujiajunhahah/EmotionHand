# 🚀 EmotionHand 快速启动指南

## 一键启动命令

### 🎭 专业版实时可视化 (推荐)
```bash
python realtime_emotion_visualizer.py
```
**功能**: 三面板专业监测 + 3D手势实时渲染

### 🎨 优化3D手势演示
```bash
python visualize_hand_3d_optimized.py
```
**功能**: 震撼3D手部模型 + 情绪颜色映射

### 🔧 个体化校准系统
```bash
python calibration_system.py
```
**功能**: 60秒快速校准，建立个人生理基线

### 🎯 智能启动器 (选择器)
```bash
python start_demo.py
```
**功能**: 图形化选择所有演示模式

---

## 🎬 演示效果预览

### 1. 专业实时可视化系统
```
┌─────────────────────────────────────────────────────┐
│         EmotionHand 实时情绪监测系统              │
├─────────────────────┬───────────────────────────────┤
│   🎭 情绪状态监测    │    📊 3D手势可视化            │
│   • 状态时间线       │    • 动态手势模型              │
│   • 置信度显示       │    • 情绪颜色映射              │
│   • 推理说明         │    • 实时数据驱动              │
├─────────────────────┼───────────────────────────────┤
│   📡 信号质量监测    │    ⚙️ 系统状态                │
│   • EMG/GSR质量曲线  │    • FPS显示                  │
│   • SNR/夹顶率       │    • 延迟监控                  │
│   • 连接状态         │    • 统计信息                  │
└─────────────────────┴───────────────────────────────┘
```

### 2. 3D手势优化演示
- ✅ 震撼的半透明3D手掌
- ✅ 渐变色手指动态弯曲
- ✅ 情绪状态驱动的颜色变化
- ✅ 背景粒子效果增强
- ✅ 15FPS流畅渲染

### 3. 个体化校准流程
```
🎯 校准过程:
1️⃣ 静息基准采集 (30秒) - 完全放松状态
2️⃣ 轻握活动采集 (30秒) - 轻握拳练习
3️⃣ 自动计算归一化参数
4️⃣ 生成个人校准档案
5️⃣ 保存供后续使用
```

---

## ⚡ 性能要求

### 最低配置
- **CPU**: 双核 2.0GHz+
- **内存**: 4GB RAM
- **Python**: 3.8+
- **依赖**: numpy, scipy, matplotlib, pandas

### 推荐配置
- **CPU**: 四核 3.0GHz+ (保证<100ms延迟)
- **内存**: 8GB RAM
- **显卡**: 支持OpenGL 3.0+ (提升3D渲染)

---

## 🔧 故障排除

### 演示启动失败
```bash
# 1. 检查Python版本
python --version

# 2. 安装依赖
pip install numpy scipy matplotlib pandas

# 3. 检查文件存在
ls realtime_emotion_visualizer.py
```

### 字体显示问题
```bash
# 已优化中文字体支持，如仍有问题：
pip install font-manager

# 或在代码中手动设置字体
# plt.rcParams['font.sans-serif'] = ['Your Font Name']
```

### 性能优化
```bash
# 如遇到延迟高：
# 1. 降低目标FPS (修改配置文件)
# 2. 减小窗口大小
# 3. 关闭其他Python进程
```

---

## 📱 快速测试

### 10秒快速体验
```bash
# 运行核心测试 (不需要可视化)
python -c "
from signal_processing_engine import RealTimeSignalProcessor
import time

processor = RealTimeSignalProcessor()
processor.start()

print('🧪 10秒信号处理测试...')
for i in range(100):
    emg = [0.1, 0.2, 0.15, 0.8]  # 模拟4通道
    gsr = 0.25
    processor.add_data(emg, gsr)
    time.sleep(0.1)

stats = processor.get_performance_stats()
print(f'✅ 测试完成: {stats[\"fps\"]:.1f} FPS, 延迟: {stats[\"latency_ms\"]:.1f}ms')
"
```

### 情绪识别快速测试
```bash
python -c "
from emotion_state_detector import EnsembleDetector
import json

with open('signal_processing_config.json') as f:
    config = json.load(f)

detector = EnsembleDetector(config)

# 测试不同情绪状态
test_cases = [
    ('放松', {'rms': 0.1, 'gsr_tonic': 0.15}),
    ('专注', {'rms': 0.4, 'gsr_tonic': 0.4}),
    ('紧张', {'rms': 0.8, 'gsr_tonic': 0.7})
]

for name, features in test_cases:
    pred = detector.predict_state(features, {}, {})
    print(f'{name}: {pred.state.value} (置信度: {pred.confidence:.2f})')
"
```

---

## 🎯 推荐启动顺序

### 首次使用
1. **校准**: `python calibration_system.py` (60秒建立个人基线)
2. **体验**: `python realtime_emotion_visualizer.py` (专业实时可视化)
3. **欣赏**: `python visualize_hand_3d_optimized.py` (震撼3D效果)

### 开发者测试
1. **引擎测试**: `python signal_processing_engine.py`
2. **检测器测试**: `python emotion_state_detector.py`
3. **集成测试**: `python start_demo.py`

---

## 🎉 即刻开始

### 最快启动方式
```bash
# 一键启动专业版
python realtime_emotion_visualizer.py

# 或使用智能选择器
python start_demo.py
```

### 查看系统状态
```bash
# 检查所有文件
ls -la *.py *.json

# 查看系统信息
python start_demo.py
```

**现在就开始体验企业级EMG+GSR信号处理的强大威力！** 🚀