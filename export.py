import torch
import os

# ==== 配置部分 ==== 
pt_path = 'C:/Users/20416/Desktop/rl_example/dqn_path_planner.pt'  # 你的pt路径
onnx_path = 'C:/Users/20416/Desktop/rl_example/dqn_path_planner.onnx'  # 输出onnx路径
min_dim = 1
max_dim = 512
opset = 11  # 可以改成13或更高（按需）

# ==== 加载 TorchScript 模型 ====
print(f"🔵 正在加载TorchScript模型：{pt_path}")
model = torch.jit.load(pt_path)
model.eval()

# ==== 自动探测输入尺寸 ====
print(f"🛠️ 开始探测输入尺寸...")
input_dim = None

for dim in range(min_dim, max_dim + 1):
    try:
        dummy_input = torch.randn(1, dim)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ 可能的输入尺寸：{dummy_input.shape}，输出尺寸：{output.shape}")
        input_dim = dim
        break  # 找到就停，不用继续了
    except Exception as e:
        continue

if input_dim is None:
    raise RuntimeError("❌ 没能找到合适的输入尺寸，请扩大搜索范围。")

# ==== 导出到 ONNX ====
print(f"🚀 开始导出 ONNX，使用输入尺寸：{input_dim}")

dummy_input = torch.randn(1, input_dim)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"🎯 导出成功！ONNX文件保存在：{onnx_path}")
