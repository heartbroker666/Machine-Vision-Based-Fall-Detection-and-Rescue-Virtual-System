"""
train_yolov8s.py
YOLOv8s 对比训练脚本（用于与 YOLOv8n 性能对比）
依赖：pip install ultralytics
"""

from ultralytics import YOLO
import torch


def main():
    # ===== 1. GPU检查 =====
    print("=== GPU 可用性检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {gpu_name}  显存: {vram:.1f} GB")
        device = "0"
        # YOLOv8s参数量更大，显存要求更高，batch相应调小
        if vram >= 10:
            batch = 16
        elif vram >= 8:
            batch = 8
        elif vram >= 6:
            batch = 4
        else:
            batch = 2
            print("[提示] 显存较小，batch自动降至2")
    else:
        print("[警告] 未检测到GPU，使用CPU训练")
        device = "cpu"
        batch = 2

    print(f"[Batch] {batch}")

    # ===== 2. 加载 YOLOv8s 模型 =====
    print("\n=== 加载 YOLOv8s 模型 ===")
    model = YOLO("yolov8s.pt")   # ← YOLOv8s 权重

    # ===== 3. 训练（参数与 YOLOv8n 保持一致，确保对比公平） =====
    print("\n=== 开始训练 YOLOv8s ===")
    results = model.train(
        data="D:/Users/Lenovo/Graduation_project/Dataset/mix_dataset/webot_real/data.yaml",
        name="fall_yolov8s_compare",   # ← 独立保存目录
        exist_ok=True,

        # ── 轮数（与YOLOv8n一致）────────────────
        epochs=50,
        patience=10,

        # ── 图像和批次 ────────────────────────────
        imgsz=640,
        batch=batch,
        workers=4,

        # ── 优化器（与YOLOv8n一致）──────────────
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=2,

        # ── 正则化 ────────────────────────────────
        momentum=0.937,
        weight_decay=0.0005,

        # ── 冻结backbone（与YOLOv8n一致）────────
        freeze=10,

        # ── 数据增强（与YOLOv8n一致）────────────
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        shear=2.0,
        perspective=0.001,
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # ── 训练配置 ──────────────────────────────
        device=device,
        pretrained=True,
        amp=True,
        fraction=1.0,
        cache=False,
        plots=True,
        save=True,
        save_period=5,
    )

    # ===== 4. 验证模型 =====
    print("\n=== 验证 YOLOv8s 模型 ===")
    metrics = model.val()

    if hasattr(metrics, 'results_dict'):
        box_metrics = metrics.results_dict
    else:
        box_metrics = {
            'metrics/precision(B)': metrics.box.mp,
            'metrics/recall(B)':    metrics.box.mr,
            'metrics/mAP50(B)':     metrics.box.map50,
            'metrics/mAP50-95(B)':  metrics.box.map,
        }

    precision = box_metrics.get('metrics/precision(B)', 0)
    recall    = box_metrics.get('metrics/recall(B)',    0)
    map50     = box_metrics.get('metrics/mAP50(B)',     0)
    map5095   = box_metrics.get('metrics/mAP50-95(B)', 0)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n✅ YOLOv8s 验证集关键指标:")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  mAP50      : {map50:.4f}")
    print(f"  mAP50-95   : {map5095:.4f}")
    print(f"\n📁 最佳模型: {model.trainer.best}")
    print(f"📁 最后模型: {model.trainer.last}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()