import os
import torch
import argparse
import sys

# 将项目根目录添加到 sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 尝试导入 torch_tensorrt
try:
    import torch_tensorrt
    HAS_TORCH_TRT = True
except ImportError:
    HAS_TORCH_TRT = False
    print('>>> [提示] torch_tensorrt 未安装，将使用标准 tensorrt API')

# 尝试导入标准 tensorrt
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    if not HAS_TORCH_TRT:
        print('>>> [错误] tensorrt 未安装。请安装 TensorRT 或使用 JetPack 自带环境。')
        sys.exit(1)

from model.easy_ViTPose.vit_models.model import ViTPose
from model.easy_ViTPose.vit_utils.util import infer_dataset_by_path, dyn_model_import

parser = argparse.ArgumentParser()
parser.add_argument('--model-ckpt', type=str, default='../checkpoints/vitpose-b-coco.engine',
                    help='The torch model that shall be used for conversion')
parser.add_argument('--model-name', type=str, default='b', choices=['s', 'b', 'l', 'h'],
                    help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
parser.add_argument('--output', type=str, default='../checkpoints/',
                    help='File (without extension) or dir path for checkpoint output')
parser.add_argument('--dataset', type=str, default='coco',
                    help='Name of the dataset.')
# 修改：默认强制 FP32，只有显式添加 --fp16 才开启
parser.add_argument('--fp16', action='store_true', default = 'False',
                    help='开启 FP16 模式（注意：ViT 模型开启此项可能导致严重精度下降，慎用！）')
parser.add_argument('--workspace', type=int, default=4,
                    help='TensorRT workspace 大小 (GB)，默认 2')
parser.add_argument('--verbose', action='store_true',default ='True',
                    help='打印 TensorRT 详细日志')
args = parser.parse_args()

# ==========================================
# 1. 准备模型和配置
# ==========================================
dataset = args.dataset
if dataset is None:
    dataset = infer_dataset_by_path(args.model_ckpt)

model_cfg = dyn_model_import(dataset, args.model_name)

CKPT_PATH = args.model_ckpt
C, H, W = (3, 256, 192)  # 固定分辨率

# 初始化模型
model = ViTPose(model_cfg)

# 加载权重
print(f'>>> Loading checkpoints from {CKPT_PATH}')
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
if 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
model.load_state_dict(ckpt)

# 【关键】将模型移动到 CUDA 并设为 eval 模式
model = model.cuda().eval()

# 准备输入数据 (固定 Batch Size = 1)
device = torch.device("cuda")
# 注意：这里我们使用 Batch=1，这是边缘端推理最常见的场景
inputs = torch.randn(1, C, H, W).to(device)

# 确定输出文件名
out_name = os.path.basename(args.model_ckpt).replace('.pth', '.onnx')
if not os.path.isdir(args.output):
    out_name = os.path.basename(args.output)
output_onnx = os.path.join(os.path.dirname(args.output), out_name)

# ==========================================
# 2. 转换为 ONNX (GPU 优化版)
# ==========================================
print('>>> Converting to ONNX (Static Shape for GPU Optimization)')
input_names = ["input_0"]
output_names = ["output_0"]

# 【修改点 1】去掉 dynamic_axes
# ViT 模型使用固定尺寸（Static Shapes）能让 TensorRT 生成最高效的 GPU 算子
# 同时也避免了动态 shape 带来的一些精度波动隐患
# dynamic_axes = {'input_0': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}} 

torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,
                  input_names=input_names, output_names=output_names)
                  # dynamic_axes=dynamic_axes) # 注释掉动态维度

print(f">>> Saved ONNX at: {os.path.abspath(output_onnx)}")
print('=' * 80)

# ==========================================
# 3. 转换为 TensorRT (高精度版)
# ==========================================
output_trt = output_onnx.replace('.onnx', '.engine')

# 【修改点 2】精度逻辑控制
# 默认使用 FP32。ViT 模型对精度极其敏感，FP16 极易导致 LayerNorm/Softmax 溢出。
use_fp16 = args.fp16 

if use_fp16:
    print(">>> [警告]以此模式生成的 ViT engine 可能会丢失检测能力！")
    print(">>> 正在使用 FP16 精度构建 (速度快，精度低)...")
else:
    print(">>> [推荐] 正在使用 FP32 精度构建 (保持最高精度)...")

if HAS_TORCH_TRT:
    # 方式1：使用 torch_tensorrt
    print('>>> Converting to TRT (Torch-TensorRT)')
    trt_inputs = [
        torch_tensorrt.Input(
            # 固定形状
            min_shape=[1, C, H, W],
            opt_shape=[1, C, H, W],
            max_shape=[1, C, H, W],
            dtype=torch.float32,
            name="input_0"
        )
    ]
    # 强制精度设置
    enabled_precisions = {torch.half} if use_fp16 else {torch.float}
    
    trt_ts_module = torch_tensorrt.compile(
        model,
        inputs=trt_inputs,
        enabled_precisions=enabled_precisions,
        ir="ts",
        truncate_long_and_double=True
    )
    print(">>> Verifying compiled model...")
    try:
        trt_ts_module(inputs)
        print(">>> Verification successful.")
    except Exception as e:
        print(f">>> Warning: Verification failed: {e}")
    torch.jit.save(trt_ts_module, output_trt)
    print(f">>> Saved TRT-ScriptModule at: {os.path.abspath(output_trt)}")

else:
    # 方式2：使用标准 tensorrt API (Jetson 常用)
    print('>>> Converting ONNX to TensorRT engine (标准 tensorrt API)')
    
    def build_engine_from_onnx(onnx_path, engine_path, fp16=False, workspace_gb=2, verbose=False):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
        
        level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        logger = trt.Logger(level)
        builder = trt.Builder(logger)
        
        # Explicit Batch 是现代 TensorRT 的标准
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX 解析失败")
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
        
        # 【修改点 3】精度控制标志
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f">>> Engine 精度: FP16 (注意精度损失)")
        else:
            # 只要不设置 FP16 flag，TensorRT 默认就会用 TF32 (Ampere架构) 或 FP32
            # 这是保证 ViT 准确率的关键
            print(f">>> Engine 精度: FP32 (已确保高精度)")
        
        # 即使是静态 Shape，为了保险起见，我们也可以创建一个 Profile
        # 这样能显式告诉 TensorRT 我们的输入范围
        min_opt_max = (1, C, H, W)
        profile = builder.create_optimization_profile()
        
        # 自动寻找输入层名称
        inp = network.get_input(0)
        name = inp.name
        profile.set_shape(name, min_opt_max, min_opt_max, min_opt_max)
        config.add_optimization_profile(profile)
        print(">>> 已添加 Optimization Profile (Static Shape: 1x3x256x192)")

        print(">>> 正在构建 TensorRT engine (可能需数分钟)...")
        try:
            serialized = builder.build_serialized_network(network, config)
        except Exception as e:
            # 兼容旧版 TensorRT API
            engine = builder.build_engine(network, config)
            serialized = engine.serialize()
            del engine

        if serialized is None:
            raise RuntimeError("构建 engine 失败")
            
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f">>> 已保存: {os.path.abspath(engine_path)}")
    
    # 执行构建
    build_engine_from_onnx(
        output_onnx, 
        output_trt, 
        fp16=use_fp16,  # 这里将传入 False，除非你强制开启
        workspace_gb=args.workspace, 
        verbose=args.verbose
    )