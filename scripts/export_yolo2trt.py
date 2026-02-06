from ultralytics import YOLO

# 加载你训练好的权重 (建议使用你重新训练过的权重，如果只是测试可以用官方的)
model = YOLO("../checkpoints/yolov8n-pose.pt")

# 导出为 TensorRT 引擎
# format="engine": 导出格式
# device=0: 使用第一张显卡 (RTX 4070)
# imgsz=128: 强制指定输入大小 (再次提醒：如果没重训过，精度会很差)
# half=True: 开启 FP16 半精度加速
model.export(format="engine", device=0, imgsz=640, half=True)