import argparse
import cv2
import numpy as np
import torch
import time
import os
import yaml
# ONNXRuntime（可选）：用于 STGCN 的 .onnx 推理
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from collections import deque
from ultralytics import YOLO

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _cfg_get(config: dict | None, path: str, default):
    """用 'a.b.c' 形式读取配置，读取不到则返回 default。"""
    if not config:
        return default
    cur = config
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _log(config: dict | None, msg: str, level: str = "INFO"):
    """轻量日志：runtime.verbose=true 时输出调试信息。"""
    verbose = bool(_cfg_get(config, "runtime.verbose", False))
    if level in ("DEBUG",) and not verbose:
        return
    print(f"[{level}] {msg}")


def _auto_output_video_path(config: dict, input_source) -> str:
    out_cfg = config.get("output", {}) if isinstance(config, dict) else {}
    p = out_cfg.get("video_path", None)
    if p:
        return p

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = "output"
    if isinstance(input_source, str) and input_source and not input_source.isdigit():
        try:
            base = os.path.splitext(os.path.basename(input_source))[0] or "output"
        except Exception:
            base = "output"
    return f"{base}_{ts}.mp4"


# ---------------------------------------------------------------------
# 跌倒变量传出说明（bool / int）
# - config.processing.fall_confirm_frames: 连续多少帧预测为跌倒才判定为确认跌倒。
# - system.fall (bool): 是否有人确认跌倒；system.fall_consecutive (int): 最大连续跌倒帧数。
# - system.get_fall_status() -> {fall, fall_consecutive, by_track}。
# - run_main(config, fall_callback=lambda s: ...): 每帧调用 fall_callback(get_fall_status()) 传出。
# ---------------------------------------------------------------------


# =====================================================================
# 1. 核心工具函数 (后处理：关键点 -> STGCN)
# =====================================================================


def self_norm(kpt, bbox):
    """
    归一化关键点坐标
    kpt: (2, T, 17, 1),  bbox: (T, 4) [x1, y1, w, h]
    返回: (2, T, 17, 1) 归一化后的坐标
    """
    # bbox格式: [x1, y1, w, h]
    tl = bbox[:, 0:2]  # 左上角坐标 (T, 2)
    wh = bbox[:, 2:]  # 宽高 (T, 2)

    # 扩展维度以匹配kpt的形状
    tl = np.expand_dims(np.transpose(tl, (1, 0)), (2, 3))  # (2, T, 1, 1)
    wh = np.expand_dims(np.transpose(wh, (1, 0)), (2, 3))  # (2, T, 1, 1)

    # 归一化: (kpt - tl) / wh，然后缩放到 (384, 512)
    res = (kpt - tl) / wh
    res *= np.expand_dims(np.array([[384.], [512.]]), (2, 3))
    return res


def convert_keypoints_to_stgcn_format(all_kpts, all_bbox, target_frames=50):
    """
    将关键点序列转换为STGCN需要的格式
    输入:
        all_kpts: list of list, 每帧的关键点 [[x, y, conf], ...] 共17个点
        all_bbox: list of list, 每帧的bbox [x1, y1, x2, y2]
        target_frames: 目标帧数，默认50
    返回:
        stgcn_input: (2, 50, 17, 1) numpy array
        scores: (50, 17, 1) numpy array
    """
    T = len(all_kpts)

    if T == 0:
        # 没有关键点数据：返回零数组（上线时不要频繁刷屏）
        return np.zeros((2, target_frames, 17, 1), dtype=np.float32), \
            np.zeros((target_frames, 17, 1), dtype=np.float32)

    # 1. 转换为numpy数组
    # all_kpts: (T, 17, 3) -> 提取 (T, 17, 2) 坐标和 (T, 17, 1) 置信度
    kpts_array = np.array(all_kpts)  # (T, 17, 3)
    all_kpts_coords = kpts_array[:, :, :2]  # (T, 17, 2) [x, y]
    all_scores = kpts_array[:, :, 2:3]  # (T, 17, 1) [conf]

    # 2. 转换bbox格式: [x1, y1, x2, y2] -> [x1, y1, w, h]
    bbox_array = np.array(all_bbox)  # (T, 4)
    bbox_formatted = np.zeros((T, 4), dtype=np.float32)
    bbox_formatted[:, 0] = bbox_array[:, 0]  # x1
    bbox_formatted[:, 1] = bbox_array[:, 1]  # y1
    bbox_formatted[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]  # w = x2 - x1
    bbox_formatted[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]  # h = y2 - y1

    # 3. 转置为 (2, T, 17, 1)
    keypoint = np.expand_dims(np.transpose(all_kpts_coords, [2, 0, 1]), -1)  # (2, T, 17, 1)

    # 4. 归一化
    keypoint = self_norm(keypoint, bbox_formatted)

    # 5. 统一帧数到target_frames
    # 策略：
    # - 超过target_frames帧: 连续截取中间target_frames帧
    # - 不足target_frames帧: 复制最后一帧进行补帧（而不是补0）
    # - 恰好target_frames帧: 无需处理
    current_frames = keypoint.shape[1]
    if current_frames > target_frames:
        # 超过target_frames帧: 连续截取中间target_frames帧
        frame_start = (current_frames - target_frames) // 2
        keypoint = keypoint[:, frame_start:frame_start + target_frames, :, :]
        all_scores = all_scores[frame_start:frame_start + target_frames, :, :]
    elif current_frames < target_frames:
        # 不足target_frames帧: 复制前面的帧进行补帧
        pad_length = target_frames - current_frames
        if current_frames > 0:
            # 策略：循环复制已有的帧，直到达到target_frames
            # 例如：如果有10帧，需要50帧，则复制为：1-10, 1-10, 1-10, 1-10, 1-10
            repeat_times = (pad_length // current_frames) + 1
            pad_kp_list = []
            pad_score_list = []

            # 复制整个序列多次
            for _ in range(repeat_times):
                pad_kp_list.append(keypoint)
                pad_score_list.append(all_scores)

            # 拼接并截取到需要的长度
            pad_kp = np.concatenate(pad_kp_list, axis=1)
            pad_score = np.concatenate(pad_score_list, axis=0)

            # 只取前面需要的部分
            pad_kp = pad_kp[:, :pad_length, :, :]
            pad_score = pad_score[:pad_length, :, :]

            keypoint = np.concatenate([keypoint, pad_kp], axis=1)
            all_scores = np.concatenate([all_scores, pad_score], axis=0)
        else:
            # 如果没有帧，补0
            keypoint = np.concatenate([
                keypoint,
                np.zeros((2, pad_length, 17, 1), dtype=keypoint.dtype)
            ], axis=1)
            all_scores = np.concatenate([
                all_scores,
                np.zeros((pad_length, 17, 1), dtype=all_scores.dtype)
            ], axis=0)
    else:
        # 恰好 target_frames：无需处理
        pass

    # 确保最终形状正确
    assert keypoint.shape == (2, target_frames, 17, 1), \
        f"关键点形状错误: {keypoint.shape}, 期望: (2, {target_frames}, 17, 1)"
    assert all_scores.shape == (target_frames, 17, 1), \
        f"置信度形状错误: {all_scores.shape}, 期望: ({target_frames}, 17, 1)"

    return keypoint, all_scores


def draw_skeleton(img, keypoints, dataset_name='coco', conf_thres=0.4, config=None, keypoint_names=None):
    """
    绘制骨架 + 显示关键点名称和置信度
    支持 COCO (17点) 和 AIC (14点)
    """
    # 使用配置或默认值
    if config is None:
        draw_config = {
            'keypoint': {'enabled': True, 'color': [0, 255, 0], 'radius': 4, 'show_conf': True,
                         'show_name': False, 'conf_threshold': 0.4, 'name_offset': [0, -15], 'conf_offset': [0, -5]},
            'skeleton': {'enabled': True, 'color': [255, 200, 0], 'thickness': 2}
        }
    else:
        draw_config = config.get('drawing', {})

    # 获取关键点配置和阈值
    kp_config = draw_config.get('keypoint', {})
    # 如果配置中有conf_threshold，优先使用配置值
    if 'conf_threshold' in kp_config:
        conf_thres = kp_config['conf_threshold']
    kp_conf_thres = conf_thres  # 统一使用这个阈值

    skeleton_links = []
    if dataset_name == 'coco':
        skeleton_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    elif dataset_name == 'aic':
        skeleton_links = [
            (12, 13), (13, 0), (13, 3),
            (0, 1), (1, 2), (3, 4), (4, 5),
            (0, 6), (3, 9), (6, 7), (7, 8), (9, 10), (10, 11)
        ]
    else:
        print(f"[警告] 未知数据集: {dataset_name}, 跳过骨架连线")

    # 绘制关键点
    if kp_config.get('enabled', True):
        kp_color = tuple(kp_config.get('color', [0, 255, 0]))
        kp_radius = kp_config.get('radius', 4)
        show_conf = kp_config.get('show_conf', True)
        show_name = kp_config.get('show_name', False)
        name_offset = kp_config.get('name_offset', [0, -15])
        conf_offset = kp_config.get('conf_offset', [0, -5])

        for idx, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > kp_conf_thres:
                # 绘制关键点
                cv2.circle(img, (int(x), int(y)), kp_radius, kp_color, -1)

                # 显示关键点名称和置信度
                if show_name and keypoint_names and idx in keypoint_names:
                    kp_name = keypoint_names[idx]
                    name_x = int(x) + name_offset[0]
                    name_y = int(y) + name_offset[1]
                    # 如果显示名称和置信度，将它们组合在一起
                    if show_conf:
                        label = f"{kp_name}:{conf:.2f}"
                    else:
                        label = kp_name
                    cv2.putText(img, label, (name_x, name_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                elif show_conf:
                    # 只显示置信度
                    conf_x = int(x) + conf_offset[0]
                    conf_y = int(y) + conf_offset[1]
                    conf_label = f"{conf:.2f}"
                    cv2.putText(img, conf_label, (conf_x, conf_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 绘制骨架连线
    skel_config = draw_config.get('skeleton', {})
    if skel_config.get('enabled', True):
        skel_color = tuple(skel_config.get('color', [255, 200, 0]))
        skel_thickness = skel_config.get('thickness', 2)

        for idx_a, idx_b in skeleton_links:
            if idx_a < len(keypoints) and idx_b < len(keypoints):
                kp_a = keypoints[idx_a]
                kp_b = keypoints[idx_b]
                if kp_a[2] > kp_conf_thres and kp_b[2] > kp_conf_thres:
                    cv2.line(img, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])),
                             skel_color, skel_thickness)
    return img


# =====================================================================
# 2. 主类: 跌倒检测系统
# =====================================================================

class FallDetectionSystem:
    def __init__(self, yolo_pose_path, stgcn_path, device='cuda', window_size=50, config=None):
        self.device = torch.device(device)
        self.window_size = window_size
        self.config = config or {}
        # 单开关：use_cuda 控制“是否全链路走 GPU”
        # - 若 use_cuda=true 且 torch.cuda.is_available(): torch 用 cuda；ORT 也优先用 CUDAExecutionProvider
        # - 若 use_cuda=false：torch 用 cpu；ORT 用 CPUExecutionProvider
        self.use_cuda = bool(self.config.get('device', {}).get('use_cuda', False)) and torch.cuda.is_available()

        # ID映射配置
        id_config = self.config.get('id_mapping', {})
        self.id_mapping_enabled = id_config.get('enabled', False)
        self.id_mapping_auto = id_config.get('auto_generate', True)
        self.id_mapping = id_config.get('mapping', {})

        # 类别映射配置
        class_mapping = self.config.get('class_mapping', {})
        self.class_names = {
            0: class_mapping.get('0', class_mapping.get(0, 'Fall')),  # 支持字符串键和整数键
            1: class_mapping.get('1', class_mapping.get(1, 'Normal'))
        }

        # COCO关键点名称映射
        self.coco_keypoint_names = {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }

        # 缓冲区倍数
        buffer_multiplier = self.config.get('processing', {}).get('buffer_multiplier', 2)

        # 二次确认：连续多少帧预测为跌倒才判定为跌倒
        self.fall_confirm_frames = max(1, int(self.config.get('processing', {}).get('fall_confirm_frames', 3)))
        self.fall_consecutive_count = {}  # {track_id: int} 连续跌倒预测次数

        # 存储每个track_id的STGCN结果（用于在检测框旁边显示）
        self.stgcn_results = {}  # {track_id: {'class': int, 'prob': float, 'conf': float, 'fall_confirmed': bool, 'fall_consecutive_count': int}}

        # 对外暴露的跌倒变量（bool / int），供外部读取
        self.fall = False  # bool: 是否有人确认跌倒
        self.fall_consecutive = 0  # int: 当前最大连续跌倒帧数（或主目标）

        # 加载 YOLOv8-Pose（单模型同时输出检测框 + 17 点关键点）
        _log(self.config, f"加载YOLOv8-Pose: {yolo_pose_path}")
        self.yolo = YOLO(yolo_pose_path)

        _log(self.config, f"加载STGCN: {stgcn_path}")
        if not os.path.exists(stgcn_path):
            raise FileNotFoundError(f"文件不存在: {stgcn_path}")

        stgcn_path_lower = stgcn_path.lower()
        self.stgcn_is_engine = stgcn_path_lower.endswith(".engine")
        self.stgcn_is_onnx = stgcn_path_lower.endswith(".onnx")

        if self.stgcn_is_engine:
            # TensorRT engine（需 tensorrt + CUDA）
            if trt is None:
                raise ImportError("加载 .engine 需要安装 tensorrt，请先安装")
            if self.device.type != "cuda":
                raise RuntimeError("STGCN .engine 仅支持 CUDA，请设置 device 为 cuda")
            logger = trt.Logger(trt.Logger.WARNING)
            with open(stgcn_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.stgcn_engine = runtime.deserialize_cuda_engine(f.read())
            if self.stgcn_engine is None:
                raise RuntimeError("STGCN engine 反序列化失败")
            self.stgcn_context = self.stgcn_engine.create_execution_context()
            if self.stgcn_context is None:
                raise RuntimeError("STGCN engine 创建 execution context 失败")
            # 约定：binding 0 为输入 (1,2,50,17,1)，binding 1 为输出 (1,num_classes)
            try:
                shp = list(self.stgcn_engine.get_binding_shape(1))
                if any(s == -1 for s in shp):
                    shp = [1, 2]  # 动态维时按 Fall/Normal 两类
                self._stgcn_out_shape = shp
            except Exception:
                self._stgcn_out_shape = [1, 2]
            _log(self.config, f"STGCN 使用 TensorRT engine，输出形状: {self._stgcn_out_shape}")
        elif self.stgcn_is_onnx:
            if ort is None:
                raise ImportError("使用 STGCN .onnx 需要安装 onnxruntime / onnxruntime-gpu")

            # 默认优先 GPU，没有 GPU 或 CUDA EP 不可用则自动回退 CPU
            providers = ["CPUExecutionProvider"]
            if self.use_cuda:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            _log(self.config, f"STGCN 使用 ONNXRuntime, providers={providers}")
            self.stgcn_session = ort.InferenceSession(stgcn_path, providers=providers)
            self.stgcn_input_name = self.stgcn_session.get_inputs()[0].name
            self.stgcn_output_name = self.stgcn_session.get_outputs()[0].name
            _log(self.config, f"STGCN 输入: {self.stgcn_input_name}, 输出: {self.stgcn_output_name}", level="DEBUG")
        else:
            raise RuntimeError("STGCN 模型仅支持 .engine 或 .onnx")

        _log(self.config, f"二次确认帧数: {self.fall_confirm_frames}（连续>=此值才判定跌倒）")
        _log(self.config, "系统就绪")

        # 关键点序列缓冲区（按track_id存储）
        self.keypoint_buffers = {}  # {track_id: {'kpts': deque, 'bboxes': deque}}
        self.track_id_counter = 0
        self.buffer_maxlen = window_size * buffer_multiplier

    def get_display_id(self, track_id):
        """获取显示用的ID（支持自动生成Person1/Person2或手动映射）"""
        if not self.id_mapping_enabled:
            return f"ID{track_id}"

        if self.id_mapping_auto:
            # 自动生成Person1, Person2等
            return f"Person{track_id + 1}"
        else:
            # 使用手动映射
            if track_id in self.id_mapping:
                return self.id_mapping[track_id]
            return f"Person{track_id + 1}"  # 如果没有映射，也使用自动生成

    def _run_stgcn(self, input_batch: np.ndarray) -> np.ndarray:
        """
        统一的 STGCN 推理入口。
        input_batch: (1, 2, 50, 17, 1) float32
        返回: (1, num_classes) float32
        """
        if getattr(self, "stgcn_is_engine", False):
            # TensorRT engine：用 GPU 上的 torch tensor 做 binding，再 execute
            inp = torch.from_numpy(input_batch).float().cuda()
            out_shape = getattr(self, "_stgcn_out_shape", [1, 2])
            out = torch.empty(out_shape, dtype=torch.float32).cuda()
            bindings = [inp.data_ptr(), out.data_ptr()]
            self.stgcn_context.execute_v2(bindings=bindings)
            return out.cpu().numpy()
        if getattr(self, "stgcn_is_onnx", False):
            outputs = self.stgcn_session.run(
                [self.stgcn_output_name],
                {self.stgcn_input_name: input_batch}
            )
            return outputs[0]
        raise RuntimeError("STGCN 未正确初始化（既不是 .engine 也不是 .onnx）")

    def _update_fall_status(self):
        """根据 stgcn_results 更新对外暴露的 fall / fall_consecutive"""
        any_confirmed = False
        max_consecutive = 0
        for r in self.stgcn_results.values():
            if r.get('fall_confirmed', False):
                any_confirmed = True
            n = r.get('fall_consecutive_count', 0)
            if n > max_consecutive:
                max_consecutive = n
        self.fall = any_confirmed
        self.fall_consecutive = max_consecutive

    def get_fall_status(self):
        """
        获取跌倒状态，供外部读取。
        返回:
            fall: bool, 是否有人确认跌倒
            fall_consecutive: int, 当前最大连续跌倒帧数
            by_track: {track_id: {'fall': bool, 'fall_consecutive': int}}
        """
        by_track = {}
        for tid, r in self.stgcn_results.items():
            by_track[tid] = {
                'fall': r.get('fall_confirmed', False),
                'fall_consecutive': r.get('fall_consecutive_count', 0)
            }
        return {
            'fall': self.fall,
            'fall_consecutive': self.fall_consecutive,
            'by_track': by_track
        }

    def process_frame(self, frame, dataset_name='coco', box_conf_thres=0.5,
                      pose_conf_thres=0.4, run_pose_estimation=True,
                      background_mode: str = "original"):
        """
        处理单帧
        返回: frame, all_keypoints, detection_results
        """
        img_h, img_w = frame.shape[:2]

        # 后台绘制底图：支持原图或纯黑背景（用于 benchmark 火柴人展示）
        if background_mode == "black":
            draw_img = np.zeros_like(frame)
        else:
            # 默认在原图副本上绘制，避免修改输入原始帧
            draw_img = frame.copy()

        # 获取绘制配置
        draw_config = self.config.get('drawing', {})
        bbox_config = draw_config.get('bbox', {})
        text_config = draw_config.get('text', {})
        min_box_size = self.config.get('detection', {}).get('min_box_size', 10)

        # 运行时模式：normal / benchmark
        runtime_mode = str(_cfg_get(self.config, "runtime.mode", "normal")).lower()
        is_benchmark = runtime_mode == "benchmark"
        hide_bbox_in_benchmark = bool(
            _cfg_get(self.config, "runtime.benchmark_hide_bbox", True)
        )

        # 1. YOLOv8-Pose 检测（同时输出检测框 + 关键点）
        results = self.yolo.predict(frame, conf=box_conf_thres, classes=0, verbose=False)
        yolo_result = results[0]
        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        confs = yolo_result.boxes.conf.cpu().numpy()

        # 关键点（全局坐标）：(N, 17, 2)，以及每个关键点的置信度 (N, 17)
        kpts_xy = None
        kpts_conf = None
        if hasattr(yolo_result, "keypoints") and yolo_result.keypoints is not None:
            try:
                kpts_xy = yolo_result.keypoints.xy.cpu().numpy()  # (N, 17, 2)
                if hasattr(yolo_result.keypoints, "conf") and yolo_result.keypoints.conf is not None:
                    kpts_conf = yolo_result.keypoints.conf.cpu().numpy()  # (N, 17)
            except Exception as e:
                print(f"[警告] 读取YOLO关键点失败: {e}")
                kpts_xy, kpts_conf = None, None

        # 按置信度排序，并限制最多处理的人数
        max_persons = self.config.get('detection', {}).get('max_persons', 1)
        order = np.argsort(-confs)  # 从高到低
        if max_persons > 0:
            order = order[:max_persons]
        boxes = boxes[order]
        confs = confs[order]

        all_keypoints = []
        current_track_ids = []

        for i, box in enumerate(boxes):
            box_score = confs[i]
            if box_score < box_conf_thres:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                continue

            # 简单的track_id分配（实际应该用MOT）
            track_id = i  # 简化处理，实际应该用跟踪算法
            display_id = self.get_display_id(track_id)

            # 绘制YOLO框（benchmark 模式下可按配置隐藏，只保留火柴人骨架）
            if bbox_config.get('enabled', True) and not (is_benchmark and hide_bbox_in_benchmark):
                # 仅当二次确认跌倒（fall_confirmed）时，检测框与文字高亮为红色
                is_fall = False
                fall_color = tuple(self.config.get('display', {}).get('result', {}).get('fall_color', [0, 0, 255]))
                normal_color = tuple(self.config.get('display', {}).get('result', {}).get('normal_color', [0, 255, 0]))
                if track_id in self.stgcn_results:
                    try:
                        is_fall = bool(self.stgcn_results[track_id].get('fall_confirmed', False))
                    except Exception:
                        is_fall = False

                bbox_color = fall_color if is_fall else tuple(bbox_config.get('color', [255, 255, 0]))
                bbox_thickness = bbox_config.get('thickness', 2)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

                # 显示Person ID和检测框置信度
                text_color = fall_color if is_fall else tuple(text_config.get('color', [255, 255, 0]))
                text_scale = text_config.get('scale', 0.5)
                text_thickness = text_config.get('thickness', 2)
                font = getattr(cv2, text_config.get('font', 'FONT_HERSHEY_SIMPLEX'))

                label_y = y1 - 10
                # Person ID
                cv2.putText(draw_img, display_id, (x1, label_y),
                            font, text_scale, text_color, text_thickness)

                # 检测框置信度
                if bbox_config.get('show_score', True):
                    box_label = f"Box:{box_score:.2f}"
                    cv2.putText(draw_img, box_label, (x1, label_y + 20),
                                font, text_scale * 0.8, text_color, text_thickness)

                # 在检测框旁边显示STGCN结果（如果有）
                if bbox_config.get('show_stgcn_result', True) and track_id in self.stgcn_results:
                    stgcn_result = self.stgcn_results[track_id]
                    class_id = stgcn_result['class']
                    class_name = self.class_names.get(class_id, f"Class{class_id}")
                    prob = stgcn_result.get('prob', stgcn_result.get('fall_prob', 0.0))
                    conf = stgcn_result.get('conf', prob)  # conf使用概率值（百分制）
                    n_consec = stgcn_result.get('fall_consecutive_count', 0)
                    confirmed = stgcn_result.get('fall_confirmed', False)

                    # 根据二次确认结果选择颜色（确认跌倒才标红）
                    if confirmed:
                        result_color = fall_color
                    else:
                        result_color = normal_color

                    offset = bbox_config.get('stgcn_result_offset', [5, 25])
                    result_x = x2 + offset[0]
                    result_y = y1 + offset[1]

                    # 显示类别、概率、置信度、连续跌倒帧数：分多行绘制（cv2.putText 不支持 \n）
                    line_gap = max(14, int(20 * text_scale))
                    cv2.putText(draw_img, f"{class_name}", (result_x, result_y),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(draw_img, f"Prob:{prob * 100:.1f}%", (result_x, result_y + line_gap),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(draw_img, f"Conf:{conf * 100:.1f}%", (result_x, result_y + 2 * line_gap),
                                font, text_scale, result_color, text_thickness)
                    cv2.putText(draw_img, f"Fall:{n_consec}/{self.fall_confirm_frames}" + (" OK" if confirmed else ""),
                                (result_x, result_y + 3 * line_gap), font, text_scale, result_color, text_thickness)

            # 姿态估计：直接使用 YOLOv8-Pose 的关键点输出
            if run_pose_estimation and kpts_xy is not None:
                idx = order[i]
                if idx < kpts_xy.shape[0]:
                    person_kpt_xy = kpts_xy[idx]  # (17, 2)
                    if kpts_conf is not None and idx < kpts_conf.shape[0]:
                        person_kpt_conf = kpts_conf[idx]  # (17,)
                    else:
                        # 若关键点置信度不可用，则使用检测框置信度作为所有关键点的置信度
                        person_kpt_conf = np.full(person_kpt_xy.shape[0], box_score, dtype=np.float32)

                    kpts = []
                    for j in range(person_kpt_xy.shape[0]):
                        x, y = person_kpt_xy[j]
                        conf_k = float(person_kpt_conf[j])
                        kpts.append([float(x), float(y), conf_k])

                    all_keypoints.append(kpts)

                    # 存储到缓冲区
                    if track_id not in self.keypoint_buffers:
                        self.keypoint_buffers[track_id] = {
                            'kpts': deque(maxlen=self.buffer_maxlen),
                            'bboxes': deque(maxlen=self.buffer_maxlen)
                        }

                    self.keypoint_buffers[track_id]['kpts'].append(kpts)
                    self.keypoint_buffers[track_id]['bboxes'].append([x1, y1, x2, y2])
                    current_track_ids.append(track_id)

                    # 绘制骨架（传入关键点名称）
                    keypoint_names = self.coco_keypoint_names if dataset_name == 'coco' else None
                    draw_skeleton(draw_img, kpts, dataset_name=dataset_name,
                                  conf_thres=pose_conf_thres, config=self.config,
                                  keypoint_names=keypoint_names)

        # 清理丢失的track_id，并重置其连续跌倒计数
        lost_ids = set(self.keypoint_buffers.keys()) - set(current_track_ids)
        for lost_id in lost_ids:
            if lost_id in self.keypoint_buffers:
                del self.keypoint_buffers[lost_id]
            if lost_id in self.stgcn_results:
                del self.stgcn_results[lost_id]
            if lost_id in self.fall_consecutive_count:
                del self.fall_consecutive_count[lost_id]
        if lost_ids:
            self._update_fall_status()

        return draw_img, all_keypoints, current_track_ids

    def predict_fall(self, track_id):
        """
        对指定track_id进行跌倒检测
        返回: {'class': 0或1, 'score': float, 'fall_prob': float}
        """
        if track_id not in self.keypoint_buffers:
            return None

        buffer = self.keypoint_buffers[track_id]
        kpts_list = list(buffer['kpts'])
        bboxes_list = list(buffer['bboxes'])

        # 获取最小预测帧数配置
        min_frames = self.config.get('processing', {}).get('min_frames_for_prediction', self.window_size)
        use_sliding = self.config.get('processing', {}).get('use_sliding_window', True)

        # 检查是否有足够的数据
        if len(kpts_list) < min_frames:
            return None  # 数据不足，不进行预测

        # 使用滑动窗口：只使用最新的window_size帧（确保实时性）
        if use_sliding and len(kpts_list) > self.window_size:
            kpts_list = kpts_list[-self.window_size:]  # 只取最新的window_size帧
            bboxes_list = bboxes_list[-self.window_size:]
            _log(self.config,
                 f"[STGCN推理] Track ID: {track_id}, 缓冲区总帧数: {len(buffer['kpts'])}, 使用最新{self.window_size}帧（滑动窗口，实时更新）",
                 level="DEBUG")
        elif use_sliding:
            # 即使不足window_size，也使用所有可用帧（会在convert_keypoints_to_stgcn_format中补帧）
            _log(self.config,
                 f"[STGCN推理] Track ID: {track_id}, 帧数: {len(kpts_list)}（不足{self.window_size}帧，将补帧）",
                 level="DEBUG")
        else:
            _log(self.config,
                 f"[STGCN推理] Track ID: {track_id}, 帧数: {len(kpts_list)}（使用全部缓冲区）",
                 level="DEBUG")

        # 转换为STGCN格式
        stgcn_input, scores = convert_keypoints_to_stgcn_format(
            kpts_list, bboxes_list, target_frames=self.window_size
        )

        # 添加batch维度: (2, 50, 17, 1) -> (1, 2, 50, 17, 1)
        stgcn_input_batch = np.expand_dims(stgcn_input, axis=0).astype(np.float32)
        _log(self.config,
             f"[STGCN推理] 输入形状: {stgcn_input_batch.shape}, "
             f"min={stgcn_input_batch.min():.4f}, max={stgcn_input_batch.max():.4f}, "
             f"mean={stgcn_input_batch.mean():.4f}",
             level="DEBUG")

        # STGCN推理（统一走 _run_stgcn，支持 ONNX / TensorRT engine）
        try:
            stgcn_out = self._run_stgcn(stgcn_input_batch)  # (1, num_classes)
            output_logit = stgcn_out[0]
            _log(self.config, f"[STGCN推理] 输出logits: {output_logit}", level="DEBUG")

            # 后处理
            # STGCN输出的是logits，需要先计算softmax得到概率
            predicted_class = int(np.argmax(output_logit))  # 使用argmax找到预测类别

            # 计算softmax概率（将logits转换为0-1之间的概率）
            exp_logits = np.exp(output_logit - np.max(output_logit))  # 数值稳定
            probs = exp_logits / np.sum(exp_logits)

            # 获取预测类别的概率（0-1之间，百分制）
            class_prob = float(probs[predicted_class])
            fall_prob = float(probs[0])  # 跌倒概率
            normal_prob = float(probs[1]) if len(probs) > 1 else 0.0  # 正常概率

            # conf使用softmax后的概率值（百分制），而不是logit值
            conf = class_prob

            # 跌倒概率底线：只有当跌倒类别概率足够高时，才参与“连续跌倒帧”计数
            fall_prob_threshold = float(self.config.get('processing', {}).get('fall_prob_threshold', 0.0))

            # 二次确认：连续 N 帧预测为跌倒才判定为确认跌倒
            if predicted_class == 0 and fall_prob >= fall_prob_threshold:  # Fall 且概率足够高
                self.fall_consecutive_count[track_id] = self.fall_consecutive_count.get(track_id, 0) + 1
            else:  # 其余情况（包括概率太低）都视为未跌倒，重置计数
                self.fall_consecutive_count[track_id] = 0
            n_consecutive = self.fall_consecutive_count[track_id]
            fall_confirmed = n_consecutive >= self.fall_confirm_frames

            result = {
                'class': predicted_class,
                'score': float(output_logit[predicted_class]),  # 原始logit值（用于调试）
                'prob': class_prob,  # 预测类别的概率（0-1）
                'conf': conf,  # 置信度（使用概率值，百分制）
                'fall_prob': fall_prob,  # 跌倒概率
                'normal_prob': normal_prob,  # 正常概率
                'all_logits': output_logit.tolist(),
                'all_probs': probs.tolist(),
                'fall_confirmed': fall_confirmed,  # bool: 是否确认跌倒
                'fall_consecutive_count': n_consecutive  # int: 连续跌倒帧数
            }

            # 存储结果以便在检测框旁边显示
            self.stgcn_results[track_id] = {
                'class': predicted_class,
                'prob': class_prob,
                'conf': conf,  # 使用概率值作为置信度
                'fall_prob': fall_prob,
                'normal_prob': normal_prob,
                'fall_confirmed': fall_confirmed,
                'fall_consecutive_count': n_consecutive
            }

            # 更新对外暴露的跌倒变量（bool / int）
            self._update_fall_status()

            class_name = self.class_names.get(predicted_class, f"Class{predicted_class}")
            _log(self.config,
                 f"STGCN: class={predicted_class}({class_name}) prob={class_prob:.3f} fall_prob={fall_prob:.3f} "
                 f"fall={n_consecutive}/{self.fall_confirm_frames} confirmed={fall_confirmed}",
                 level="DEBUG")
            return result
        except Exception as e:
            _log(self.config, f"STGCN推理错误: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return None


def run_main(config, fall_callback=None):
    """
    视频/摄像头模式。
    fall_callback: 可选，每帧调用 callback(get_fall_status()) 传出跌倒变量，便于外部告警等。
    """
    device = 'cuda' if (config['device']['use_cuda'] and torch.cuda.is_available()) else 'cpu'
    system = FallDetectionSystem(
        config['model']['yolo_pose'],
        config['model']['stgcn'],
        device=device,
        window_size=config['processing']['window_size'],
        config=config
    )

    # 打开视频源（上线版：收敛为通用 OpenCV 打开方式）
    input_source = config['input']['source']
    _log(config, f"打开视频源: {input_source}")

    if isinstance(input_source, str) and input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {input_source}")
        print(f"[错误] 请检查:")
        print(f"  - 摄像头是否已连接并正常工作")
        print(f"  - 摄像头ID是否正确（尝试 0 或 1）")
        print(f"  - 视频文件路径是否正确")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 读取第一帧验证
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"[错误] 无法从视频源读取帧")
        print(f"[错误] 摄像头可能被其他程序占用或无法正常工作")
        cap.release()
        return

    _log(config, f"读取首帧成功，尺寸: {test_frame.shape[1]}x{test_frame.shape[0]}")

    # 获取视频属性（使用实际读取的帧尺寸，避免阻塞）
    width = test_frame.shape[1]
    height = test_frame.shape[0]

    # 尝试获取FPS（如果失败则使用默认值）
    try:
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0:
            fps_video = 30
            print(f"[视频源] 无法获取FPS，使用默认值: {fps_video}")
    except:
        fps_video = 30
        print(f"[视频源] 获取FPS失败，使用默认值: {fps_video}")

    _log(config, f"开始推理: {width}x{height}, FPS={fps_video:.1f}")
    _log(config, f"阈值: box={config['detection']['box_conf_threshold']}, pose={config['detection']['pose_conf_threshold']}")
    _log(config, f"窗口: {config['processing']['window_size']} 帧")
    _log(config, "按 'q' 退出")

    frame_count = 0
    last_prediction_frame = {}

    # 运行模式与 benchmark 配置
    runtime_mode = str(_cfg_get(config, "runtime.mode", "normal")).lower()
    is_benchmark = runtime_mode == "benchmark"
    benchmark_show_image = bool(_cfg_get(config, "runtime.benchmark_show_image", False))
    benchmark_skeleton_only = bool(_cfg_get(config, "runtime.benchmark_skeleton_only", True))
    benchmark_log_interval = int(_cfg_get(config, "runtime.benchmark_log_interval", 60))

    # benchmark 统计：区分摄像头 FPS 与模型 FPS
    bench_start_time = time.time()
    cam_frame_count = 0
    model_frame_count = 0

    # 结果视频保存（可选）
    # 规则：只在输入是“视频文件”（后缀 .mp4/.avi）时保存；摄像头不保存
    is_video_file = (
        isinstance(input_source, str)
        and input_source.lower().endswith((".mp4", ".avi"))
        and os.path.isfile(input_source)
    )
    save_video = bool(_cfg_get(config, "output.save_video", False)) and is_video_file
    writer = None
    if save_video:
        out_path = _auto_output_video_path(config, input_source)
        out_fps_cfg = _cfg_get(config, "output.video_fps", None)
        out_fps = float(out_fps_cfg) if out_fps_cfg is not None else float(fps_video if fps_video > 0 else 30.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))
        if not writer.isOpened():
            _log(config, f"无法创建视频写入器: {out_path}（将不保存视频）", level="ERROR")
            writer = None
        else:
            _log(config, f"保存预测视频到: {out_path} (fps={out_fps:.1f})")

    print(f"[开始推理] 进入主循环，开始处理帧...\n")

    last_cam_time = time.time()

    while True:
        t0 = time.time()

        # 摄像头读取计时（用于统计摄像头 FPS）
        cam_t_start = time.time()
        ret, frame = cap.read()
        cam_t_end = time.time()

        if not ret:
            print(f"[警告] 无法读取帧（可能视频结束或摄像头断开），退出")
            break

        if frame is None or frame.size == 0:
            print(f"[警告] 读取到空帧，跳过")
            continue

        frame_count += 1
        cam_frame_count += 1

        rotate = config['input']['rotate']
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 跳帧逻辑
        pose_interval = config['processing']['pose_interval']
        run_pose = (frame_count % pose_interval == 0)

        # 处理帧（统计模型推理时间，用于模型 FPS）
        model_t_start = time.time()
        background_mode = "black" if (is_benchmark and benchmark_skeleton_only) else "original"
        res_img, current_kpts, track_ids = system.process_frame(
            frame,
            box_conf_thres=config['detection']['box_conf_threshold'],
            pose_conf_thres=config['detection']['pose_conf_threshold'],
            run_pose_estimation=run_pose,
            background_mode=background_mode,
        )
        model_t_end = time.time()
        model_frame_count += 1

        # 对每个track_id进行跌倒检测（当数据足够时）
        window_size = config['processing']['window_size']
        prediction_interval = config['processing']['prediction_interval']
        min_frames = config['processing'].get('min_frames_for_prediction', window_size)

        for track_id in track_ids:
            if track_id in system.keypoint_buffers:
                buffer = system.keypoint_buffers[track_id]
                buffer_len = len(buffer['kpts'])

                # 检查是否有足够的数据进行预测
                if buffer_len >= min_frames:
                    # 每N帧预测一次（避免频繁预测）
                    if track_id not in last_prediction_frame or \
                            (frame_count - last_prediction_frame[track_id]) >= prediction_interval:
                        result = system.predict_fall(track_id)
                        if result:
                            last_prediction_frame[track_id] = frame_count
                            # STGCN结果已在 process_frame 中显示；fall/fall_consecutive 已更新
                else:
                    # 显示等待状态
                    if buffer_len > 0:
                        display_id = system.get_display_id(track_id)
                        progress = (buffer_len / window_size) * 100
                        # 可选：在检测框旁边显示收集进度
                        pass  # 暂时不显示，避免界面混乱

        # 实时 FPS（整条管线）
        fps_real = 1.0 / (time.time() - t0 + 1e-5)

        # benchmark 模式下，统计并打印 摄像头 FPS & 模型 FPS
        if is_benchmark:
            elapsed = max(time.time() - bench_start_time, 1e-5)
            cam_fps = cam_frame_count / elapsed
            model_fps = model_frame_count / elapsed

            # 每隔指定帧数打印一次基准信息，包含关键置信度
            if frame_count % max(1, benchmark_log_interval) == 0:
                status = system.get_fall_status()
                max_fall_prob = 0.0
                for r in system.stgcn_results.values():
                    try:
                        p = float(r.get("fall_prob", 0.0))
                    except Exception:
                        p = 0.0
                    if p > max_fall_prob:
                        max_fall_prob = p

                print(
                    f"[BENCH] frame={frame_count} "
                    f"cam_fps={cam_fps:.2f} "
                    f"model_fps={model_fps:.2f} "
                    f"max_fall_prob={max_fall_prob:.3f} "
                    f"fall={status['fall']} "
                    f"fall_consecutive={status['fall_consecutive']}"
                )

        # UI显示（上线版：只保留 FPS 一行，可通过 runtime.show_fps 控制）
        if bool(_cfg_get(config, "runtime.show_fps", True)):
            cv2.putText(res_img, f"FPS: {fps_real:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # 控制台打印实时FPS（每隔一定帧数，避免刷屏）
            if frame_count % 30 == 0:
                _log(config, f"实时 FPS: {fps_real:.1f}")

        # 保存视频帧（写入预测结果图像）
        if writer is not None:
            try:
                writer.write(res_img)
            except Exception as e:
                _log(config, f"写视频帧失败: {e}", level="ERROR")
                try:
                    writer.release()
                except Exception:
                    pass
                writer = None

        # 传出跌倒变量：每帧调用 fall_callback（若提供）
        if fall_callback is not None:
            try:
                fall_callback(system.get_fall_status())
            except Exception as e:
                pass  # 避免回调异常影响主循环

        # 是否显示窗口
        show_window = bool(_cfg_get(config, "output.show_result", True))
        # benchmark 模式下允许单独关闭图像显示
        if is_benchmark and not benchmark_show_image:
            show_window = False

        if show_window:
            window_name = config.get('output', {}).get('window_name', 'Fall Detection')
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, res_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # 不显示窗口时，仍允许 Ctrl+C 退出；这里不阻塞
            pass

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    _log(config, "推理结束")
    return system


def load_config(config_path='config.yaml'):
    """
    加载配置文件，提供完整的默认配置
    """
    # 最小默认配置（上线版：尽量少、尽量稳定）
    default_config = {
        'model': {
            'yolo_pose': './checkpoints/yolov8n-pose.engine',
            'stgcn': './checkpoints/stgcn.engine'
        },
        'input': {
            'source': '0',
            'mode': 'auto',
            'rotate': 0
        },
        'output': {
            'show_result': False,
            'window_name': 'Fall Detection'
        },
        'class_mapping': {
            '0': 'Fall',
            '1': 'Normal'
        },
        'detection': {
            'box_conf_threshold': 0.5,
            'pose_conf_threshold': 0.4,
            'min_box_size': 10,
            'max_persons': 1
        },
        'processing': {
            'pose_interval': 1,
            'window_size': 50,
            'prediction_interval': 10,
            'buffer_multiplier': 2,
            'min_frames_for_prediction': 30,
            'use_sliding_window': True,
            'fall_confirm_frames': 3  # 二次确认：连续多少帧预测为跌倒才判定为跌倒（>=1）
        },
        'device': {
            'use_cuda': True,
        },
        # 运行时参数（上线默认不刷屏）
        'runtime': {
            # 是否输出详细调试信息（DEBUG）
            'verbose': False,
            # 是否在图像上叠加 FPS 文本
            'show_fps': True,
            # 运行模式: normal / benchmark
            'mode': 'normal',
            # benchmark 模式下是否仍然弹出图像窗口
            'benchmark_show_image': False,
            # benchmark 模式下是否仅显示黑底火柴人骨架
            'benchmark_skeleton_only': True,
            # benchmark 模式下是否隐藏检测框
            'benchmark_hide_bbox': True,
            # benchmark 模式下控制台打印统计信息的帧间隔
            'benchmark_log_interval': 60,
        }
    }

    if not os.path.exists(config_path):
        print(f"[警告] 配置文件 {config_path} 不存在，使用默认配置")
        return default_config

    with open(config_path, 'r', encoding='utf-8') as f:
        file_config = yaml.safe_load(f)

    # 兼容历史键名：models -> model（上线版仍保留一次转换，避免部署时踩坑）
    if 'models' in file_config and 'model' not in file_config:
        file_config['model'] = file_config.pop('models')

    # 深度合并配置（文件配置覆盖默认配置）
    def deep_merge(default, override):
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    config = deep_merge(default_config, file_config)

    # 处理模型路径：相对路径优先相对 config 文件目录
    config_dir = os.path.dirname(os.path.abspath(config_path)) or '.'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for model_key in ['yolo_pose', 'stgcn']:
        if model_key in config.get('model', {}):
            model_path = config['model'][model_key]
            if model_path and not os.path.isabs(model_path):
                # 相对路径：先尝试相对于配置文件目录，再尝试相对于脚本目录
                if not os.path.exists(model_path):
                    # 尝试1：相对于配置文件目录
                    alt_path1 = os.path.join(config_dir, model_path.lstrip('./'))
                    if os.path.exists(alt_path1):
                        config['model'][model_key] = alt_path1
                    else:
                        # 尝试2：相对于脚本目录
                        alt_path2 = os.path.join(script_dir, model_path.lstrip('./'))
                        if os.path.exists(alt_path2):
                            config['model'][model_key] = alt_path2

    # 基础校验（上线版：尽早失败）
    if not config.get('model', {}).get('yolo_pose'):
        raise ValueError("配置缺少 model.yolo_pose")
    if not config.get('model', {}).get('stgcn'):
        raise ValueError("配置缺少 model.stgcn")

    return config


def determine_mode(config):
    """
    根据配置确定运行模式
    """
    input_source = config['input']['source']
    mode = config['input']['mode']

    if mode == 'video':
        return 'video'
    elif mode == 'camera':
        return 'video'
    elif mode == 'auto':
        # 自动识别
        if isinstance(input_source, str) and input_source.isdigit():
            return 'video'  # 摄像头
        return 'video'  # 上线版统一走视频模式（文件/摄像头都算 video）
    else:
        return 'video'  # 默认视频模式


if __name__ == "__main__":
    # 支持命令行参数（可选，优先级高于配置文件）
    parser = argparse.ArgumentParser(description='跌倒检测推理系统')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='配置文件路径（默认: config.yaml）')
    parser.add_argument('--input', type=str, default=None,
                        help='覆盖配置文件中的输入源')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 如果命令行指定了input，覆盖配置
    if args.input:
        config['input']['source'] = args.input

    # 检查STGCN路径
    if not config['model']['stgcn']:
        print("[错误] 配置文件中必须指定STGCN模型路径")
        exit(1)

    # 确定运行模式
    mode = determine_mode(config)

    _log(config, f"使用配置文件: {args.config}")
    _log(config, f"运行模式: {mode}")
    _log(config, f"输入源: {config['input']['source']}")

    run_main(config)
