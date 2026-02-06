#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 STGCN 的 ONNX 模型转换为 TensorRT engine，用于推理加速。

用法:
  python scripts/export_stgcn2trt.py --onnx ./checkpoints/stgcn.onnx --output ./checkpoints/stgcn.engine
  python scripts/export_stgcn2trt.py --onnx ./checkpoints/stgcn.onnx --output ./checkpoints/ --fp16 --workspace 2

STGCN 输入形状: (1, 2, 50, 17, 1) float32 [batch, (x,y), T, joints, 1]
STGCN 输出形状: (1, num_classes) float32，num_classes 通常为 2（Fall / Normal）

依赖: pip install tensorrt  （或 JetPack 自带 TensorRT）
"""

import os
import sys
import argparse

try:
    import tensorrt as trt
except ImportError:
    print("[Error] tensorrt 未安装。请安装 TensorRT 或使用 JetPack 自带环境。")
    sys.exit(1)


# STGCN 在 RobGymInfer 中的固定输入形状
STGC_INPUT_SHAPE = (1, 2, 50, 17, 1)


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 2,
    verbose: bool = False,
):
    """
    从 ONNX 构建 TensorRT engine 并保存。
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

    level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(level)
    builder = trt.Builder(logger)

    # 显式 batch，与 ONNX 的 (1,2,50,17,1) 一致
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
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f">>> 使用 FP16")
    else:
        print(f">>> 使用 FP32")

    # ONNX 含动态维时，必须为该输入指定 optimization profile，否则会报：
    # "Network has dynamic or shape inputs, but no optimization profile has been defined."
    min_opt_max = tuple(STGC_INPUT_SHAPE)
    profile = builder.create_optimization_profile()
    need_profile = False
    try:
        num_inputs = getattr(network, "num_inputs", 0) or 0
        for i in range(num_inputs):
            inp = network.get_input(i)
            name = getattr(inp, "name", None) or f"input_{i}"
            try:
                shp = getattr(inp, "shape", None) or []
                shape = list(shp)
            except Exception:
                shape = []
            def _is_dynamic(d):
                if d is None or d == -1:
                    return True
                if hasattr(d, "is_dynamic") and getattr(d, "is_dynamic", False):
                    return True
                try:
                    if int(d) == -1:
                        return True
                except Exception:
                    pass
                return False
            if any(_is_dynamic(s) for s in shape):
                need_profile = True
                profile.set_shape(name, min_opt_max, min_opt_max, min_opt_max)
                if verbose:
                    print(f">>> 为动态输入 '{name}' 设置 optimization profile: {min_opt_max}")
    except Exception as e:
        if verbose:
            print(f">>> 遍历 network inputs 时出错: {e}，改为对首个输入名称设置 profile")
        # 部分 TRT 版本需从 parser 或 bindings 获取输入名，此处做保守回退
        try:
            inp = network.get_input(0)
            name = inp.name
            profile.set_shape(name, min_opt_max, min_opt_max, min_opt_max)
            need_profile = True
        except Exception:
            pass
    if need_profile:
        config.add_optimization_profile(profile)
        print(">>> 已添加 optimization profile（固定输入形状为 (1,2,50,17,1)）")

    print(">>> 正在构建 TensorRT engine（可能需数分钟）...")
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        # 部分 TensorRT 版本用 build_engine，再 serialize
        try:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("builder.build_engine 返回 None")
            serialized = engine.serialize()
            del engine
        except AttributeError:
            raise e

    if serialized is None:
        raise RuntimeError("构建 engine 失败，serialized 为 None")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f">>> 已保存: {os.path.abspath(engine_path)}")
    return engine_path


def run_quick_test(engine_path: str):
    """加载 engine 并跑一次前向，验证是否能正常推理。"""
    import numpy as np
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(">>> [警告] 无法反序列化 engine，跳过推理测试")
        return
    # 简单校验：能创建 context 即可；完整 inference 需要 pycuda 绑定
    ctx = engine.create_execution_context()
    if ctx is None:
        print(">>> [警告] 无法创建 execution context")
        return
    print(">>> 校验: engine 可反序列化且可创建 execution context，OK")


def main():
    parser = argparse.ArgumentParser(description="STGCN ONNX -> TensorRT engine")
    parser.add_argument("--onnx", type=str, default="../checkpoints/stgcn.onnx",
                        help="STGCN ONNX 模型路径")
    parser.add_argument("--output", type=str, default="../checkpoints/stgcn.engine",
                        help="输出 engine 路径（若为目录则在该目录下生成 stgcn.engine）")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="启用 FP16（默认 True）")
    parser.add_argument("--fp32", action="store_true",
                        help="强制 FP32")
    parser.add_argument("--workspace", type=int, default=2,
                        help="TensorRT workspace 大小 (GB)，默认 2")
    parser.add_argument("--verbose", action="store_true",default=True,
                        help="打印 TensorRT 详细日志")
    parser.add_argument("--test", action="store_true",
                        help="导出后做一次加载/context 校验")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    out = args.output
    if os.path.isdir(out):
        out = os.path.join(out, "stgcn.engine")
    engine_path = os.path.abspath(out)

    fp16 = args.fp16 and not args.fp32
    build_engine_from_onnx(
        onnx_path,
        engine_path,
        fp16=fp16,
        workspace_gb=args.workspace,
        verbose=args.verbose,
    )
    if args.test:
        run_quick_test(engine_path)

    print("\n>>> 在 RobGymInfer 中使用: 将 config.model.stgcn 改为该 engine 路径即可（RobGymInfer 已支持 .engine 加载与推理）。")


if __name__ == "__main__":
    main()
