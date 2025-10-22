#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 演示系统启动器
一键启动所有演示
"""

import os
import sys
import subprocess
import time

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """打印启动器横幅"""
    print("""
🎭 EmotionHand 专业情绪手势识别系统 v2.0
==================================================

🚀 企业级EMG+GSR信号处理 + 3D实时可视化

特性:
• 专业信号处理: 信号→时间窗→归一化
• 个体化校准: 60秒快速适配
• 实时监测: <100ms延迟, 15-30 FPS
• 震撼可视化: 3D手势 + 质量面板
• 企业级架构: SOLID原则, 配置驱动
""")

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")

    required = ['numpy', 'scipy', 'matplotlib', 'pandas']
    missing = []

    for lib in required:
        try:
            __import__(lib)
            print(f"  ✅ {lib}")
        except ImportError:
            missing.append(lib)
            print(f"  ❌ {lib} (缺失)")

    if missing:
        print(f"\n📦 安装缺失依赖:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("✅ 所有依赖已就绪")
    return True

def get_demo_options():
    """获取演示选项"""
    return {
        '1': {
            'name': '专业实时可视化系统',
            'script': 'realtime_emotion_visualizer.py',
            'description': '三面板专业监测: 情绪状态 + 信号质量 + 3D手势',
            'features': ['实时情绪识别', '信号质量监测', '3D手势渲染', '性能监控']
        },
        '2': {
            'name': '优化3D手势演示',
            'script': 'visualize_hand_3d_optimized.py',
            'description': '震撼3D手部模型 + 实时数据驱动',
            'features': ['半透明手掌', '渐变手指', '情绪颜色映射', '15FPS渲染']
        },
        '3': {
            'name': '个体化校准系统',
            'script': 'calibration_system.py',
            'description': '60秒快速校准，建立个人生理基线',
            'features': ['30秒静息校准', '30秒活动校准', '分位归一化', '校准档案管理']
        },
        '4': {
            'name': '信号处理引擎测试',
            'script': 'signal_processing_engine.py',
            'description': '测试核心信号处理功能',
            'features': ['EMG滤波测试', 'GSR特征提取', '质量监测', '性能统计']
        },
        '5': {
            'name': '情绪检测器测试',
            'script': 'emotion_state_detector.py',
            'description': '测试规则基线情绪识别算法',
            'features': ['4种情绪状态', '置信度评估', '平滑处理', '状态统计']
        },
        '6': {
            'name': '数据采集工具',
            'script': 'data_collector.py',
            'description': '采集真实EMG+GSR训练数据',
            'features': ['实时数据采集', '手动标注', '特征提取', 'CSV导出']
        }
    }

def print_demo_options():
    """打印演示选项"""
    demos = get_demo_options()

    print("📋 可用演示:")
    print()

    for key, demo in demos.items():
        print(f"  {key}. {demo['name']}")
        print(f"     {demo['description']}")
        print(f"     特性: {', '.join(demo['features'])}")
        print()

def run_demo(choice):
    """运行选定的演示"""
    demos = get_demo_options()

    if choice not in demos:
        print("❌ 无效选择")
        return False

    demo = demos[choice]
    script = demo['script']

    # 检查脚本是否存在
    if not os.path.exists(script):
        print(f"❌ 脚本不存在: {script}")
        return False

    print(f"🚀 启动 {demo['name']}...")
    print(f"📝 执行脚本: {script}")
    print("💡 按 Ctrl+C 停止演示")
    print()

    try:
        # 运行脚本
        subprocess.run([sys.executable, script], check=True)
    except KeyboardInterrupt:
        print("\n🔚 用户中断演示")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 演示运行失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        return False

    return True

def show_system_info():
    """显示系统信息"""
    print("💻 系统信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  操作系统: {os.name}")
    print(f"  当前目录: {os.getcwd()}")

    # 检查文件
    files = [
        'realtime_emotion_visualizer.py',
        'visualize_hand_3d_optimized.py',
        'signal_processing_engine.py',
        'calibration_system.py',
        'signal_processing_config.json'
    ]

    print("\n📁 核心文件检查:")
    for file in files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")

def main():
    """主函数"""
    clear_screen()
    print_banner()

    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        sys.exit(1)

    print()
    show_system_info()
    print()

    while True:
        print_demo_options()
        print("0. 退出")
        print()

        choice = input("请选择演示 (0-6): ").strip()

        if choice == '0':
            print("👋 再见！")
            break
        elif choice in ['1', '2', '3', '4', '5', '6']:
            print()
            success = run_demo(choice)
            if not success:
                input("\n按回车键继续...")
            print()
            clear_screen()
            print_banner()
        else:
            print("❌ 无效选择，请输入 0-6")
            time.sleep(1)
            clear_screen()
            print_banner()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户退出")
    except Exception as e:
        print(f"\n❌ 启动器错误: {e}")
        input("按回车键退出...")