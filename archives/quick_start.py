#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 快速启动脚本
一键运行各种演示和功能

使用方法:
python quick_start.py [选项]

选项:
  realtime    - 运行实时数据流可视化
  demo        - 运行3D动画演示
  static      - 生成静态演示图片
  view        - 打开演示查看器
  install     - 安装项目依赖
  test        - 运行测试
  status      - 检查项目状态
  help        - 显示帮助信息
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """检查关键依赖"""
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'pandas',
        'scikit-learn', 'lightgbm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")

    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("运行 'python quick_start.py install' 来安装依赖")
        return False

    print("\n✅ 所有依赖已安装")
    return True

def install_dependencies():
    """安装项目依赖"""
    print("🔧 安装EmotionHand项目依赖...")
    print("这可能需要几分钟时间...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("\n✅ 依赖安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 依赖安装失败: {e}")
        print("\n💡 手动安装命令:")
        print("pip install -r requirements.txt")
        return False

def run_realtime_demo():
    """运行实时演示"""
    print("🚀 启动实时EMG+GSR可视化演示...")
    print("💡 提示: 按's'录制数据，按'q'退出")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, "realtime_emotion_plot.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 实时演示启动失败")
        return False
    except KeyboardInterrupt:
        print("\n👋 演示已停止")
    return True

def run_3d_demo():
    """运行3D动画演示"""
    print("🎨 启动3D手部模型动画演示...")
    print("💡 提示: 关闭窗口返回主菜单")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, "visualize_hand_demo.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 3D演示启动失败")
        return False
    except KeyboardInterrupt:
        print("\n👋 演示已停止")
    return True

def run_static_demo():
    """运行静态演示"""
    print("📊 生成静态演示图片...")
    print("💡 这将生成两个PNG演示文件")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, "hand_demo_static.py"], check=True)
        print("\n✅ 静态演示完成！")
        print("📁 生成的文件:")
        if os.path.exists("EmotionHand_Hand_Model_Demo.png"):
            print("  • EmotionHand_Hand_Model_Demo.png")
        if os.path.exists("EmotionHand_Signal_Analysis_Demo.png"):
            print("  • EmotionHand_Signal_Analysis_Demo.png")
        return True
    except subprocess.CalledProcessError:
        print("❌ 静态演示失败")
        return False

def run_demo_viewer():
    """运行演示查看器"""
    print("🎮 启动演示查看器...")
    print("💡 提示: 在菜单中选择要查看的演示")
    print("-" * 50)

    try:
        subprocess.run([sys.executable, "view_demos.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 演示查看器启动失败")
        return False
    except KeyboardInterrupt:
        print("\n👋 查看器已关闭")
    return True

def check_project_status():
    """检查项目状态"""
    print("🔍 EmotionHand项目状态检查")
    print("=" * 50)

    # 检查Python版本
    print("\n📋 Python环境:")
    check_python_version()

    # 检查依赖
    print("\n📋 依赖包:")
    deps_ok = check_dependencies()

    # 检查核心文件
    print("\n📋 核心文件:")
    core_files = [
        "realtime_emotion_plot.py",
        "visualize_hand_demo.py",
        "hand_demo_static.py",
        "view_demos.py",
        "arduino_emotion_hand.ino",
        "requirements.txt",
        "README_OPTIMIZED.md"
    ]

    for file in core_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"  ✅ {file} ({size:.1f}KB)")
        else:
            print(f"  ❌ {file}")

    # 检查演示文件
    print("\n📋 演示文件:")
    demo_files = [
        "EmotionHand_Hand_Model_Demo.png",
        "EmotionHand_Signal_Analysis_Demo.png"
    ]

    for file in demo_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024 / 1024
            print(f"  ✅ {file} ({size:.1f}MB)")
        else:
            print(f"  ⚪ {file} (可通过静态演示生成)")

    # 检查scripts目录
    print("\n📋 后端脚本:")
    if os.path.exists("scripts"):
        scripts = list(Path("scripts").glob("*.py"))
        for script in scripts:
            print(f"  ✅ scripts/{script.name}")
    else:
        print("  ❌ scripts/ 目录")

    # 检查Unity脚本
    print("\n📋 Unity脚本:")
    unity_dir = Path("unity/Assets/Scripts")
    if unity_dir.exists():
        unity_scripts = list(unity_dir.glob("*.cs"))
        for script in unity_scripts:
            print(f"  ✅ unity/Assets/Scripts/{script.name}")
    else:
        print("  ⚪ unity/ 目录 (可选)")

    # 总结
    print("\n" + "=" * 50)
    if deps_ok:
        print("🎉 项目状态良好，可以正常运行！")
        print("\n🚀 推荐的下一步:")
        print("  python quick_start.py realtime  # 运行实时演示")
        print("  python quick_start.py demo      # 运行3D演示")
    else:
        print("⚠️ 需要先安装依赖包")
        print("  python quick_start.py install   # 安装依赖")

    return deps_ok

def run_tests():
    """运行测试"""
    print("🧪 运行EmotionHand测试套件...")

    # 检查关键脚本语法
    test_files = [
        "realtime_emotion_plot.py",
        "visualize_hand_demo.py",
        "hand_demo_static.py",
        "view_demos.py"
    ]

    all_passed = True
    for file in test_files:
        if os.path.exists(file):
            try:
                # 编译检查语法
                with open(file, 'r', encoding='utf-8') as f:
                    compile(f.read(), file, 'exec')
                print(f"  ✅ {file}")
            except SyntaxError as e:
                print(f"  ❌ {file}: 语法错误 - {e}")
                all_passed = False
            except Exception as e:
                print(f"  ⚠️ {file}: {e}")
                all_passed = False
        else:
            print(f"  ❌ {file}: 文件不存在")
            all_passed = False

    if all_passed:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")

    return all_passed

def show_help():
    """显示帮助信息"""
    print("🎭 EmotionHand 快速启动脚本")
    print("=" * 50)
    print("使用方法:")
    print("  python quick_start.py [选项]")
    print("\n可用选项:")
    print("  realtime    运行实时EMG+GSR数据流可视化")
    print("  demo        运行3D手部模型动画演示")
    print("  static      生成静态演示图片")
    print("  view        打开演示查看器")
    print("  install     安装项目依赖包")
    print("  test        运行语法检查和测试")
    print("  status      检查项目完整状态")
    print("  help        显示此帮助信息")
    print("\n示例:")
    print("  python quick_start.py realtime  # 运行实时演示")
    print("  python quick_start.py install   # 安装依赖")
    print("  python quick_start.py status    # 检查状态")
    print("\n更多信息请查看:")
    print("  • README_OPTIMIZED.md - 完整项目文档")
    print("  • CODE_COMPLETE.md - 所有源代码")
    print("  • FINAL_DEMO_SUMMARY.md - 项目总结")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    print("🎭 EmotionHand 快速启动工具")
    print("=" * 50)

    commands = {
        'realtime': run_realtime_demo,
        'demo': run_3d_demo,
        'static': run_static_demo,
        'view': run_demo_viewer,
        'install': install_dependencies,
        'test': run_tests,
        'status': check_project_status,
        'help': show_help
    }

    if command in commands:
        try:
            commands[command]()
        except KeyboardInterrupt:
            print("\n\n👋 操作已取消")
        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
    else:
        print(f"❌ 未知命令: {command}")
        print("\n可用命令:", ", ".join(commands.keys()))
        print("运行 'python quick_start.py help' 查看详细帮助")

if __name__ == "__main__":
    main()