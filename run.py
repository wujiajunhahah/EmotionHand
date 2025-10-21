#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 一键启动脚本
提供便捷的系统启动和管理功能
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🎭 EmotionHand - 基于EMG+GSR的情绪状态识别系统")
    print("=" * 60)
    print("✨ 特性:")
    print("   • EMG + GSR 双模态信号融合")
    print("   • 实时推理延迟 <100ms")
    print("   • 2分钟个性化校准")
    print("   • Unity 3D实时可视化")
    print("   • 支持跨人泛化")
    print("=" * 60)

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ 需要Python 3.7或更高版本")
        return False
    else:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 检查必要的包
    required_packages = [
        'numpy', 'pandas', 'scipy', 'scikit-learn',
        'lightgbm', 'matplotlib', 'seaborn', 'joblib'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    # 检查目录结构
    required_dirs = ['scripts', 'models', 'data', 'unity', 'docs']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ 目录 {dir_name}/")
        else:
            print(f"⚠️ 目录 {dir_name}/ 不存在，将自动创建")
            os.makedirs(dir_name, exist_ok=True)

    print("✅ 环境检查完成\n")
    return True

def run_demo(mode='interactive'):
    """运行演示"""
    print("🚀 启动演示系统...")

    demo_script = os.path.join('scripts', 'demo.py')

    if not os.path.exists(demo_script):
        print(f"❌ 演示脚本不存在: {demo_script}")
        return False

    try:
        if mode == 'full':
            cmd = [sys.executable, demo_script, '--full']
        else:
            cmd = [sys.executable, demo_script, '--interactive']

        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 演示运行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")
        return True

def run_training():
    """运行模型训练"""
    print("🧠 启动模型训练...")

    training_script = os.path.join('scripts', 'training.py')

    if not os.path.exists(training_script):
        print(f"❌ 训练脚本不存在: {training_script}")
        return False

    try:
        cmd = [sys.executable, training_script]
        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        return True

def run_data_collection():
    """运行数据采集"""
    print("📊 启动数据采集...")

    collection_script = os.path.join('scripts', 'data_collection.py')

    if not os.path.exists(collection_script):
        print(f"❌ 采集脚本不存在: {collection_script}")
        return False

    try:
        cmd = [sys.executable, collection_script]
        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 数据采集失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断数据采集")
        return True

def run_calibration():
    """运行个性化校准"""
    print("⚙️ 启动个性化校准...")

    calibration_script = os.path.join('scripts', 'calibration.py')

    if not os.path.exists(calibration_script):
        print(f"❌ 校准脚本不存在: {calibration_script}")
        return False

    try:
        cmd = [sys.executable, calibration_script]
        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 校准失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断校准")
        return True

def run_real_time_inference():
    """运行实时推理"""
    print("⚡ 启动实时推理...")

    inference_script = os.path.join('scripts', 'real_time_inference.py')

    if not os.path.exists(inference_script):
        print(f"❌ 推理脚本不存在: {inference_script}")
        return False

    try:
        cmd = [sys.executable, inference_script]
        subprocess.run(cmd, check=True)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ 实时推理失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断实时推理")
        return True

def install_dependencies():
    """安装依赖包"""
    print("📦 安装项目依赖...")

    requirements_file = 'requirements.txt'
    environment_file = 'environment.yml'

    # 优先使用conda
    if os.path.exists(environment_file):
        try:
            print("使用Conda环境...")
            subprocess.run(['conda', 'env', 'create', '-f', environment_file], check=True)
            print("✅ Conda环境创建完成")
            print("请运行: conda activate emotionhand")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Conda安装失败: {e}")

    # 使用pip
    if os.path.exists(requirements_file):
        try:
            print("使用pip安装...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
            print("✅ pip安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ pip安装失败: {e}")
            return False
    else:
        print("❌ 未找到依赖文件")
        return False

def setup_project():
    """设置项目"""
    print("🔧 设置项目...")

    # 创建必要的目录
    dirs_to_create = [
        'data/public',
        'data/private/raw',
        'data/private/processed',
        'models/calibration',
        'docs/results',
        'docs/figures',
        'utils',
        'unity/Assets/Scripts',
        'unity/Assets/Materials',
        'unity/Assets/Scenes'
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

    # 创建.gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# 数据文件
data/private/raw/*.csv
data/private/processed/*.pkl
models/*.joblib
models/*.pkl

# 日志文件
*.log

# Unity
Library/
Temp/
Logs/
usersettings/

# 系统文件
.DS_Store
Thumbs.db
"""

    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ 创建 .gitignore")

    print("✅ 项目设置完成")

def show_status():
    """显示系统状态"""
    print("📊 系统状态:")
    print("-" * 40)

    # 检查模型文件
    model_files = [
        'models/gesture_lightgbm.joblib',
        'models/state_lightgbm.joblib',
        'models/scaler.joblib',
        'models/label_encoder.joblib'
    ]

    print("📁 模型文件:")
    for model_file in model_files:
        status = "✅" if os.path.exists(model_file) else "❌"
        print(f"   {status} {model_file}")

    # 检查数据文件
    data_dir = 'data'
    if os.path.exists(data_dir):
        total_files = 0
        for root, dirs, files in os.walk(data_dir):
            total_files += len(files)
        print(f"📁 数据文件: {total_files} 个文件")
    else:
        print("📁 数据文件: ❌ 数据目录不存在")

    # 检查Unity文件
    unity_files = [
        'unity/Assets/Scripts/UdpReceiver.cs',
        'unity/Assets/Scripts/EmotionHandVisualizer.cs',
        'unity/Assets/Scripts/CalibrationUI.cs'
    ]

    print("📁 Unity脚本:")
    for unity_file in unity_files:
        status = "✅" if os.path.exists(unity_file) else "❌"
        print(f"   {status} {unity_file}")

    print("-" * 40)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EmotionHand 一键启动脚本')
    parser.add_argument('command', nargs='?', choices=[
        'demo', 'train', 'collect', 'calibrate', 'inference',
        'install', 'setup', 'status'
    ], help='要执行的命令')
    parser.add_argument('--mode', choices=['full', 'interactive'],
                       default='interactive', help='演示模式')
    parser.add_argument('--skip-check', action='store_true',
                       help='跳过环境检查')

    args = parser.parse_args()

    print_banner()

    # 环境检查
    if not args.skip_check:
        if not check_environment():
            print("❌ 环境检查失败，请解决问题后重试")
            return

    # 执行命令
    if args.command == 'demo':
        run_demo(args.mode)
    elif args.command == 'train':
        run_training()
    elif args.command == 'collect':
        run_data_collection()
    elif args.command == 'calibrate':
        run_calibration()
    elif args.command == 'inference':
        run_real_time_inference()
    elif args.command == 'install':
        install_dependencies()
    elif args.command == 'setup':
        setup_project()
    elif args.command == 'status':
        show_status()
    else:
        # 交互式菜单
        interactive_menu()

def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 EmotionHand 主菜单:")
        print("1. 🚀 运行演示")
        print("2. 🧠 训练模型")
        print("3. 📊 数据采集")
        print("4. ⚙️ 个性化校准")
        print("5. ⚡ 实时推理")
        print("6. 📦 安装依赖")
        print("7. 🔧 项目设置")
        print("8. 📊 查看状态")
        print("9. 🚪 退出")

        choice = input("\n请选择操作 (1-9): ").strip()

        if choice == '1':
            mode = input("演示模式 (full/interactive) [默认: interactive]: ").strip()
            if mode not in ['full', 'interactive']:
                mode = 'interactive'
            run_demo(mode)
        elif choice == '2':
            run_training()
        elif choice == '3':
            run_data_collection()
        elif choice == '4':
            run_calibration()
        elif choice == '5':
            run_real_time_inference()
        elif choice == '6':
            install_dependencies()
        elif choice == '7':
            setup_project()
        elif choice == '8':
            show_status()
        elif choice == '9':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重试")

        input("\n按回车继续...")

if __name__ == "__main__":
    main()