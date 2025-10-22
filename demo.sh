#!/usr/bin/env bash
set -e

# EmotionHand 一键启动脚本

echo ">>> EmotionHand 演示环境自检"

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python"
    exit 1
fi

# 检查依赖
python - <<'PY'
import importlib, sys
need = ["numpy","scipy","matplotlib","pandas","pyserial","lightgbm","joblib","pyarrow"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("缺少依赖：", missing)
    print("请执行：pip install " + " ".join(missing))
    sys.exit(1)
print("✅ 依赖检查通过")
PY

if [ $? -ne 0 ]; then
    echo "❌ 依赖检查失败"
    echo "请执行：pip install numpy scipy matplotlib pandas pyserial lightgbm joblib pyarrow"
    exit 1
fi

echo "✅ 依赖OK"
echo ""
echo ">>> 启动专业实时可视化系统（可 Ctrl+C 退出）"
python realtime_emotion_visualizer.py