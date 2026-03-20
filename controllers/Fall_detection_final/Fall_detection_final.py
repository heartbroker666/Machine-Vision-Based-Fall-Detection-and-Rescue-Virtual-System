"""
Fall_detection_final.py
完善版跌倒检测系统（支持中文显示 + Emitter通知PR2救援）
支持从环境变量读取参数（配合 fall_detection_ui.py 一键启动）
"""

from controller import Supervisor
import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════
#  中文字体初始化
# ══════════════════════════════════════════════
def _load_font(size):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        try:
            f = ImageFont.truetype(path, size)
            print(f"[字体] 已加载: {path} size={size}")
            return f
        except:
            continue
    print("[字体] 未找到中文字体，使用默认字体")
    return ImageFont.load_default()

FONT_LG = _load_font(28)
FONT_MD = _load_font(22)
FONT_SM = _load_font(18)

def put_text(img_bgr, text, pos, color=(255,255,255), font=None):
    if font is None:
        font = FONT_MD
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(img_rgb)
    draw      = ImageDraw.Draw(pil_img)
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ══════════════════════════════════════════════
#  配置参数（优先读环境变量，fallback到默认值）
# ══════════════════════════════════════════════
def _env(key, default):
    return os.environ.get(key, default)

MODEL_PATH          = _env("FD_MODEL",    "D:/pythonProject/毕设3/runs/detect/fall_mixed_v1/weights/best.pt")
CAMERA_NAME         = _env("FD_CAM",      "cam0")
EMITTER_NAME        = _env("FD_EMITTER",  "fall_emitter")
OUTPUT_DIR          = _env("FD_OUTDIR",   "D:/Users/Lenovo/Graduation_project/detection_output")
CONF_THRESH         = float(_env("FD_CONF",     "0.5"))
FALL_CONFIRM_FRAMES = int(  _env("FD_FRAMES",   "15"))
ALARM_COOLDOWN      = float(_env("FD_COOLDOWN", "8.0"))

print(f"[配置] 模型:    {MODEL_PATH}")
print(f"[配置] 摄像头:  {CAMERA_NAME}")
print(f"[配置] 置信度:  {CONF_THRESH}")
print(f"[配置] 确认帧:  {FALL_CONFIRM_FRAMES}")
print(f"[配置] 冷却时间:{ALARM_COOLDOWN}s")
print(f"[配置] 输出目录:{OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
#  初始化
# ══════════════════════════════════════════════
model = YOLO(MODEL_PATH)
print(f"[模型] 加载完成，类别: {model.names}")

robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice(CAMERA_NAME)
camera.enable(timestep)
W   = camera.getWidth()
H   = camera.getHeight()
FPS = max(1, 1000 // timestep)
print(f"[摄像头] {W}x{H} @ {FPS}fps")

# 行人节点（Supervisor直接读坐标）
pedestrian_node = robot.getFromDef("PEDESTRIAN")
if pedestrian_node:
    print("[Supervisor] 已获取行人节点 PEDESTRIAN")
else:
    print("[警告] 未找到PEDESTRIAN节点，将使用像素坐标估算")

# Emitter
emitter = robot.getDevice(EMITTER_NAME)
if emitter:
    emitter.setChannel(1)
    print(f"[Emitter] 已就绪，channel=1")
else:
    print(f"[警告] 未找到Emitter '{EMITTER_NAME}'")

# 视频录制
ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(OUTPUT_DIR, f"demo_{ts}.mp4")
writer     = cv2.VideoWriter(video_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              FPS, (W, H))

# 事件日志
log_path   = os.path.join(OUTPUT_DIR, f"fall_log_{ts}.csv")
log_file   = open(log_path, "w", newline="", encoding="utf-8-sig")
log_writer = csv.writer(log_file)
log_writer.writerow(["序号","时间","置信度","世界X","世界Y","帧号"])

# ══════════════════════════════════════════════
#  状态变量
# ══════════════════════════════════════════════
fall_counter    = 0
alarm_active    = False
last_alarm_time = 0.0
alarm_count     = 0
frame_count     = 0
pr2_dispatched  = False
pr2_target      = None

print("\n[系统] 检测开始运行\n")

# ══════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════
while robot.step(timestep) != -1:
    frame_count += 1

    raw = camera.getImage()
    if raw is None or len(raw) == 0:
        continue
    try:
        img   = np.frombuffer(raw, np.uint8).reshape((H, W, 4))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except (ValueError, cv2.error):
        continue

    # ── YOLO推理 ──────────────────────────────
    results = model(frame, conf=CONF_THRESH, verbose=False)

    display            = frame.copy()
    is_fall_this_frame = False
    best_conf          = 0.0
    best_box           = None

    for box in results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = (0, 0, 255) if cls == 0 else (0, 200, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{model.names[cls]} {conf:.2f}",
                    (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls == 0:
            is_fall_this_frame = True
            if conf > best_conf:
                best_conf = conf
                best_box  = (x1, y1, x2, y2)

    # ── 连续帧确认 ────────────────────────────
    if is_fall_this_frame:
        fall_counter = min(fall_counter + 1, FALL_CONFIRM_FRAMES + 5)
    else:
        fall_counter = max(0, fall_counter - 1)

    fall_confirmed = fall_counter >= FALL_CONFIRM_FRAMES

    # ── 报警 + 通知PR2 ────────────────────────
    current_time = time.time()

    if fall_confirmed:
        if not alarm_active or (current_time - last_alarm_time) > ALARM_COOLDOWN:
            alarm_active    = True
            last_alarm_time = current_time
            alarm_count    += 1
            pr2_dispatched  = False

        if not pr2_dispatched and best_box:
            if pedestrian_node:
                pos = pedestrian_node.getPosition()
                wx  = pos[0]
                wy  = pos[1]
                print(f"[Supervisor] 行人坐标: ({wx:.2f}, {wy:.2f})")
            else:
                cx = (best_box[0] + best_box[2]) // 2
                cy = (best_box[1] + best_box[3]) // 2
                wx = (cx / W - 0.5) * 9.9
                wy = (cy / H - 0.5) * 6.6
                print(f"[像素估算] 坐标: ({wx:.2f}, {wy:.2f})")

            pr2_target     = (wx, wy)
            pr2_dispatched = True

            if emitter:
                msg = f"{wx:.4f},{wy:.4f}"
                emitter.send(msg.encode('utf-8'))

            log_writer.writerow([alarm_count,
                                  datetime.now().strftime("%H:%M:%S"),
                                  f"{best_conf:.3f}",
                                  f"{wx:.2f}", f"{wy:.2f}", frame_count])
            log_file.flush()
            print(f"[⚠ 报警 #{alarm_count}] 置信度={best_conf:.3f}  位置=({wx:.2f},{wy:.2f})")
            print(f"[→ PR2] 已发送救援坐标")

    else:
        if (current_time - last_alarm_time) > ALARM_COOLDOWN:
            alarm_active   = False
            pr2_dispatched = False

    # ── UI绘制 ────────────────────────────────
    if fall_confirmed and alarm_active:
        cv2.rectangle(display, (0, 0), (W, 58), (0, 0, 180), -1)
        display = put_text(display,
                           f"⚠ 跌倒报警！第{alarm_count}次",
                           (10, 8), (255, 255, 255), FONT_LG)
        cv2.rectangle(display, (2, 2), (W-2, H-2), (0, 0, 255), 3)
        if pr2_target:
            wx, wy = pr2_target
            display = put_text(display,
                               f"PR2救援机器人已派遣  目标:({wx:.1f},{wy:.1f})",
                               (10, H-32), (0, 220, 255), FONT_MD)
    elif is_fall_this_frame:
        cv2.rectangle(display, (0, 0), (W, 58), (0, 100, 200), -1)
        display = put_text(display,
                           f"检测中... ({fall_counter}/{FALL_CONFIRM_FRAMES}帧确认)",
                           (10, 8), (255, 255, 255), FONT_LG)
    else:
        cv2.rectangle(display, (0, 0), (W, 58), (20, 100, 20), -1)
        display = put_text(display,
                           "正常  监控运行中",
                           (10, 8), (255, 255, 255), FONT_LG)

    display = put_text(display, f"报警: {alarm_count}次",
                       (W-130, 8),  (200, 200, 200), FONT_SM)
    display = put_text(display, f"帧: {frame_count}",
                       (W-130, 30), (150, 150, 150), FONT_SM)
    if pr2_dispatched:
        display = put_text(display, "PR2: 救援中",
                           (W-130, 52), (0, 200, 255), FONT_SM)

    cv2.imshow("跌倒检测系统", display)
    writer.write(display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
log_file.close()
cv2.destroyAllWindows()
print(f"\n共处理{frame_count}帧，累计报警{alarm_count}次")
print(f"演示视频: {video_path}")
print(f"事件日志: {log_path}")