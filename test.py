# ==========================================================
#     src/test.py (终极 C++ 模块测试脚本)
# ==========================================================
import sys
from pathlib import Path
import traceback

# --- 步骤 1: 精确地告诉 Python 在哪里能找到我们编译的 .so 文件 ---
# 我们知道 .so 文件在 build-final 文件夹里
build_dir = Path(__file__).parent / "build-final"
if not build_dir.exists():
    print(f"❌ 致命错误: 'build-final' 文件夹不存在于当前目录！")
    print(f"   请确保您已经成功编译了 C++ 模块。")
    sys.exit(1)

# 将这个文件夹添加到 Python 的“搜索路径”中
sys.path.append(str(build_dir))

print(f"✅ [测试脚本] 已将路径 '{build_dir}' 添加到搜索列表。")
print(">> [测试脚本] 现在，尝试导入 'my_bridge' 模块...")

try:
    # 这是关键的一步，它会去加载 .so 文件
    import my_bridge

    print("✅✅✅ [测试脚本] 巨大成功！'my_bridge' C++ 模块已成功导入！")
except ImportError as e:
    print("\n❌ 致命错误: 导入 'my_bridge' 失败。")
    print("   这几乎总是因为操作系统找不到核心的 'libllama.so' 引擎。")
    print("   请确保您使用了正确的启动命令，其中必须包含 LD_LIBRARY_PATH。")
    print(f"   原始错误信息: {e}")
    sys.exit(1)

# --- 步骤 2: 定义模型路径 (【【【 请务必确认这个路径是正确的！ 】】】) ---
MODEL_PATH = "/mnt/data/model/Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf"

if not Path(MODEL_PATH).exists():
    print(f"\n❌ 致命错误: 在路径 '{MODEL_PATH}' 中找不到模型文件。")
    print("   请修改脚本中的 MODEL_PATH 变量，指向您真实的大模型文件。")
    sys.exit(1)

# --- 步骤 3: 初始化并运行我们的 C++ 引擎 ---
try:
    print("\n>> [测试脚本] 准备初始化 C++ LlamaEngine...")
    print("   (这可能需要一些时间来加载模型到内存/显存中)")

    # 调用 C++ 代码来创建引擎实例
    engine = my_bridge.LlamaEngine(model_path=MODEL_PATH, n_gpu_layers=99, n_ctx=4096)
    print("✅ [测试脚本] C++ 引擎初始化成功！模型已加载。")

    prompt = "Hello, world! Please write a single sentence in English."
    print(f"\n>> [测试脚本] 正在向 C++ 引擎发送提示: '{prompt}'")

    # 调用我们最核心的 generate 函数
    response = engine.generate(prompt, max_tokens=30)

    # --- 如果您能看到下面的输出，就代表我们彻底成功了 ---
    print("\n" + "=" * 60)
    print("    🎉🎉🎉 终 极 胜 利 ! C++ 引 擎 已 成 功 运 行 ! 🎉🎉🎉")
    print("=" * 60)
    print(f"\nC++ 引擎的回复是:\n\n   >>> {response} <<<\n")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ 致命错误: 在运行 C++ 引擎时发生意外。")
    print(f"   这可能是模型文件损坏，或 C++ 代码内部存在运行时问题。")
    print(f"   错误信息: {e}")
    traceback.print_exc()
    sys.exit(1)
