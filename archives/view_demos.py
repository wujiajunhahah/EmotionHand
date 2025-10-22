#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionHand 演示查看器
快速查看所有演示效果和运行状态
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def print_banner():
    """打印横幅"""
    print("=" * 70)
    print("🎭 EmotionHand 演示查看器")
    print("=" * 70)
    print("📋 可用的演示和文档:")
    print("  1. 查看静态图片演示")
    print("  2. 运行实时动画演示")
    print("  3. 查看演示文档")
    print("  4. 检查项目状态")
    print("  5. 退出")
    print("=" * 70)

def show_static_demos():
    """显示静态演示图片"""
    print("\n🖼️  静态演示图片:")

    demo_files = [
        ("EmotionHand_Hand_Model_Demo.png", "3D手部模型演示"),
        ("EmotionHand_Signal_Analysis_Demo.png", "信号分析演示")
    ]

    for filename, description in demo_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024 / 1024  # MB
            print(f"  ✅ {filename} ({file_size:.1f}MB) - {description}")

            # 尝试打开图片 (macOS)
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", filename], check=True)
                    print(f"     📱 已在默认图片查看器中打开")
                elif sys.platform == "win32":  # Windows
                    os.startfile(filename)
                    print(f"     📱 已在默认图片查看器中打开")
                else:  # Linux
                    subprocess.run(["xdg-open", filename], check=True)
                    print(f"     📱 已在默认图片查看器中打开")
            except Exception as e:
                print(f"     ⚠️ 无法自动打开: {e}")
        else:
            print(f"  ❌ {filename} - 文件不存在")

def run_realtime_demo():
    """运行实时演示"""
    print("\n⚡ 实时演示选项:")
    print("  1. 运行静态演示 (快速预览)")
    print("  2. 运行实时动画演示 (完整功能)")
    print("  3. 返回主菜单")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == '1':
        print("\n🚀 启动静态演示...")
        try:
            subprocess.run([sys.executable, "hand_demo_static.py"], check=True)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断演示")
        except Exception as e:
            print(f"❌ 演示运行失败: {e}")

    elif choice == '2':
        print("\n🚀 启动实时动画演示...")
        print("💡 提示: 关闭演示窗口返回主菜单")
        try:
            subprocess.run([sys.executable, "visualize_hand_demo.py"], check=True)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断演示")
        except Exception as e:
            print(f"❌ 演示运行失败: {e}")

    elif choice == '3':
        return
    else:
        print("❌ 无效选择")

def show_documentation():
    """显示文档"""
    print("\n📚 项目文档:")

    doc_files = [
        ("README.md", "项目主文档"),
        ("PROJECT_SUMMARY.md", "技术总结"),
        ("DEMO_SHOWCASE.md", "演示展示文档"),
        ("GITHUB_UPLOAD_GUIDE.md", "GitHub上传指南")
    ]

    for filename, description in doc_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"  ✅ {filename} ({file_size:.1f}KB) - {description}")
        else:
            print(f"  ❌ {filename} - 文件不存在")

    # 询问是否查看详细文档
    choice = input("\n是否查看演示展示文档? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        try:
            if os.path.exists("DEMO_SHOWCASE.md"):
                with open("DEMO_SHOWCASE.md", 'r', encoding='utf-8') as f:
                    content = f.read()
                print("\n" + "="*60)
                print("📋 演示展示文档内容:")
                print("="*60)
                print(content[:2000] + "..." if len(content) > 2000 else content)
            else:
                print("❌ 演示展示文档不存在")
        except Exception as e:
            print(f"❌ 读取文档失败: {e}")

def check_project_status():
    """检查项目状态"""
    print("\n🔍 项目状态检查:")

    # 检查核心文件
    core_files = [
        ("run.py", "一键启动脚本"),
        ("requirements.txt", "Python依赖"),
        ("LICENSE", "MIT许可证"),
        (".gitignore", "Git忽略规则")
    ]

    print("\n📁 核心文件:")
    for filename, description in core_files:
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"  {status} {filename} - {description}")

    # 检查脚本目录
    print("\n📂 Python脚本:")
    script_dir = Path("scripts")
    if script_dir.exists():
        scripts = list(script_dir.glob("*.py"))
        for script in scripts:
            print(f"  ✅ scripts/{script.name}")
    else:
        print("  ❌ scripts/ 目录不存在")

    # 检查Unity脚本
    print("\n🎮 Unity脚本:")
    unity_dir = Path("unity/Assets/Scripts")
    if unity_dir.exists():
        unity_scripts = list(unity_dir.glob("*.cs"))
        for script in unity_scripts:
            print(f"  ✅ unity/Assets/Scripts/{script.name}")
    else:
        print("  ❌ unity/Assets/Scripts/ 目录不存在")

    # 检查演示文件
    print("\n🎨 演示文件:")
    demo_files = [
        "visualize_hand_demo.py",
        "hand_demo_static.py",
        "EmotionHand_Hand_Model_Demo.png",
        "EmotionHand_Signal_Analysis_Demo.png"
    ]

    for filename in demo_files:
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"  {status} {filename}")

    # Git状态
    print("\n📊 Git状态:")
    try:
        result = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            modified = len([line for line in result.stdout.split('\n') if line.strip()])
            if modified == 0:
                print("  ✅ 工作区干净 (无未提交更改)")
            else:
                print(f"  ⚠️ 有 {modified} 个未提交的更改")
        else:
            print("  ❌ 不是Git仓库")
    except Exception:
        print("  ❌ 无法检查Git状态")

def main():
    """主函数"""
    while True:
        print_banner()

        choice = input("请选择操作 (1-5): ").strip()

        if choice == '1':
            show_static_demos()
        elif choice == '2':
            run_realtime_demo()
        elif choice == '3':
            show_documentation()
        elif choice == '4':
            check_project_status()
        elif choice == '5':
            print("\n👋 再见! 感谢使用 EmotionHand 演示查看器!")
            break
        else:
            print("❌ 无效选择，请重试")

        input("\n按回车键继续...")

if __name__ == "__main__":
    main()