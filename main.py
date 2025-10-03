# -*- coding: utf-8 -*-
"""
Windows 实时推理 + 资源 HUD 一体脚本
Pipeline: YOLOv11-LA (Ultralytics) + DeepSORT + RiceLCNN + Morphology
功能：实时 FPS/时延显示；CPU/RAM；GPU 利用率/显存/温度/功耗（NVML）；视频录制

注意：
- DeepSORT / RiceLCNN 的 import 与权重路径，请按你的项目结构调整。
- 若无 NVIDIA GPU 或未安装 NVML/驱动，GPU 信息将显示为 N/A。
"""

import os
import cv2
import time
import torch
import psutil
import hashlib
import numpy as np
import subprocess
import re
import shutil
from PIL import Image

# YOLO
from ultralytics import YOLO

# DeepSORT（按你的工程路径）
from deep_sort.deep_sort import DeepSort

# RiceLCNN（按你的工程路径）
from RiceLCNN.image_predict import predict_image_from_numpy

# （可选）Otsu 工具
# from Otsu import get_min_area_rect


# ===================== Windows 友好设置 =====================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ===================== 系统监控（Windows 优先 NVML） =====================
class SystemMonitor:
    """
    优先 NVML；失败则用 nvidia-smi；再失败用 PyTorch 进程显存。
    自动选取与 torch 当前设备一致的 GPU 索引。
    """
    def __init__(self):
        self.nvml_ok = False
        self.nvml = None
        self.handle = None
        self.smi_ok = shutil.which("nvidia-smi") is not None

        # 选定 PyTorch 设备索引（若可用）
        self.torch_cuda = torch.cuda.is_available()
        self.dev_index = None
        self.cuda_total_gb = None
        self.cuda_name = None
        if self.torch_cuda:
            try:
                self.dev_index = torch.cuda.current_device()
            except Exception:
                self.dev_index = 0

        # 尝试 NVML，并与 PyTorch 设备对齐
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            self.nvml = nvml

            # 优先按 PyTorch 设备索引获取 NVML 句柄
            if self.dev_index is not None:
                try:
                    self.handle = nvml.nvmlDeviceGetHandleByIndex(int(self.dev_index))
                except Exception:
                    self.handle = None

            # 如果上一步失败，枚举 NVML 设备并尽量匹配名称
            if self.handle is None:
                cnt = nvml.nvmlDeviceGetCount()
                self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
                # 若 torch 有设备名，尝试名称匹配
                if self.torch_cuda:
                    try:
                        self.cuda_name = torch.cuda.get_device_name(self.dev_index)
                        for i in range(cnt):
                            h = nvml.nvmlDeviceGetHandleByIndex(i)
                            n = nvml.nvmlDeviceGetName(h).decode("utf-8", "ignore")
                            if self.cuda_name and self.cuda_name.lower() in n.lower():
                                self.handle = h
                                self.dev_index = i
                                break
                    except Exception:
                        pass

            # 记录总显存（NVML）
            if self.handle is not None:
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.cuda_total_gb = mem.total / (1024**3)
                self.nvml_ok = True
        except Exception:
            self.nvml_ok = False

        # 若 NVML 拿不到总显存，用 PyTorch 属性兜底
        if self.cuda_total_gb is None and self.torch_cuda:
            try:
                props = torch.cuda.get_device_properties(0 if self.dev_index is None else self.dev_index)
                self.cuda_total_gb = props.total_memory / (1024**3)
                self.cuda_name = props.name
            except Exception:
                self.cuda_total_gb = None

    def _read_nvidia_smi(self):
        """使用 nvidia-smi 查询指定索引的显存/利用率/温度/功耗。"""
        if not self.smi_ok:
            return {}
        index = 0 if self.dev_index is None else int(self.dev_index)
        cmd = [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=0.4).decode("utf-8", "ignore").strip()
            # 形如: "1234, 8192, 5, 45, 20.15"
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 5:
                used_mb  = float(parts[0]); total_mb = float(parts[1])
                util_pct = float(parts[2]); temp_c   = float(parts[3])
                power_w  = None
                try:
                    power_w = float(parts[4])
                except Exception:
                    power_w = None
                return dict(
                    gpu_util=util_pct,
                    gpu_mem_used_gb=used_mb/1024.0,
                    gpu_mem_total_gb=total_mb/1024.0,
                    gpu_temp_c=temp_c,
                    gpu_power_w=power_w
                )
        except Exception:
            pass
        return {}

    def read(self):
        # CPU/RAM
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        # 先 NVML
        gpu_util = None
        gpu_mem_used_gb = None
        gpu_mem_total_gb = self.cuda_total_gb
        gpu_temp = None
        gpu_power_w = None

        if self.nvml_ok and self.handle is not None:
            try:
                ur = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = float(ur.gpu)
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                gpu_mem_used_gb = mem.used / (1024**3)
                # 覆盖总显存（NVML 更权威）
                gpu_mem_total_gb = mem.total / (1024**3)
                gpu_temp = float(self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU))
                try:
                    gpu_power_w = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                except Exception:
                    gpu_power_w = None
            except Exception:
                pass

        # 次选 nvidia-smi
        if gpu_util is None and self.smi_ok:
            smi = self._read_nvidia_smi()
            if smi:
                gpu_util = smi.get("gpu_util", None)
                gpu_mem_used_gb = smi.get("gpu_mem_used_gb", None)
                gpu_mem_total_gb = smi.get("gpu_mem_total_gb", gpu_mem_total_gb)
                gpu_temp = smi.get("gpu_temp_c", gpu_temp)
                gpu_power_w = smi.get("gpu_power_w", gpu_power_w)

        # 最后兜底：仅进程显存
        cuda_alloc_gb = None
        cuda_reserved_gb = None
        if self.torch_cuda:
            try:
                idx = 0 if self.dev_index is None else self.dev_index
                torch.cuda.set_device(idx)
                cuda_alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
                cuda_reserved_gb = torch.cuda.memory_reserved(idx) / (1024**3)
            except Exception:
                pass

        return dict(
            cpu_percent=cpu,
            ram_percent=ram,
            gpu_util=gpu_util,
            gpu_mem_used_gb=gpu_mem_used_gb,
            gpu_mem_total_gb=gpu_mem_total_gb,
            cuda_alloc_gb=cuda_alloc_gb,
            cuda_reserved_gb=cuda_reserved_gb,
            gpu_temp_c=gpu_temp,
            gpu_power_w=gpu_power_w
        )



def draw_hud(frame, fps, latency_ms, sysinfo, origin=(10, 80)):
    x, y = origin
    lines = [
        f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms",
        f"CPU: {sysinfo['cpu_percent']:.0f}%  |  RAM: {sysinfo['ram_percent']:.0f}%"
    ]

    # GPU 总显存（卡级）与进程显存
    gpu_line = "GPU: N/A"
    if sysinfo["gpu_util"] is not None:
        gpu_line = f"GPU: {sysinfo['gpu_util']:.0f}%"

    vram_line = ""
    if sysinfo["gpu_mem_used_gb"] is not None and sysinfo["gpu_mem_total_gb"] is not None:
        vram_line = f" | VRAM(used/total): {sysinfo['gpu_mem_used_gb']:.2f}/{sysinfo['gpu_mem_total_gb']:.2f} GB"

    proc_line = ""
    if sysinfo["cuda_alloc_gb"] is not None:
        if sysinfo.get("cuda_reserved_gb") is not None:
            proc_line = f" | CUDA(proc): alloc {sysinfo['cuda_alloc_gb']:.2f} / reserved {sysinfo['cuda_reserved_gb']:.2f} GB"
        else:
            proc_line = f" | CUDA(proc): alloc {sysinfo['cuda_alloc_gb']:.2f} GB"

    extra = []
    if sysinfo["gpu_temp_c"] is not None:  extra.append(f"{sysinfo['gpu_temp_c']:.0f}°C")
    if sysinfo["gpu_power_w"] is not None: extra.append(f"{sysinfo['gpu_power_w']:.1f} W")
    extra_str = ("  |  " + " / ".join(extra)) if extra else ""

    lines.append(gpu_line + vram_line + proc_line + extra_str)

    # 背景+文字
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_h = 22
    pad = 8
    panel_w = 0
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, font, font_scale, thickness)
        panel_w = max(panel_w, tw)
    panel_h = line_h * len(lines) + pad * 2
    cv2.rectangle(overlay, (x - pad, y - pad - line_h), (x + panel_w + pad, y - pad - line_h + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(frame, s, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)



# ===================== 初始化函数 =====================
def init_yolo(model_path: str, device: torch.device):
    model = YOLO(model_path)
    model.to(device)
    return model


def init_deepsort(reid_weights: str, max_dist: float, min_confidence: float,
                  nms_max_overlap: float, max_iou_distance: float, max_age: int,
                  n_init: int, nn_budget: int, num_target: int) -> DeepSort:
    deepsort = DeepSort(
        reid_weights, max_dist, min_confidence, nms_max_overlap,
        max_iou_distance, max_age, n_init, nn_budget, num_target
    )
    return deepsort


def init_webcam(camera_index=0):
    """Windows 优先用 CAP_DSHOW，避免延迟与黑屏。"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index = {camera_index})")
    return cap


def get_webcam_info(cap):
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if not frame_rate or frame_rate <= 0:
        frame_rate = 30.0
    if width == 0 or height == 0:
        width, height = 1280, 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return frame_rate, width, height


def init_video_writer(output_path: str, fps: float, frame_size: tuple):
    """Windows 上推荐 mp4v；若失败可改 'XVID' 或 'MJPG'。"""
    ensure_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not vw.isOpened():
        # 回退
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return vw


def generate_colors(class_dict):
    colors = {}
    for name in class_dict:
        name_str = str(name)
        hash_object = hashlib.md5(name_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        b = (hash_int & 0xFF0000) >> 16
        g = (hash_int & 0x00FF00) >> 8
        r = hash_int & 0x0000FF
        colors[name] = (int(b), int(g), int(r))
    return colors


# ===================== 核心逐帧处理 =====================
def process_frame(frame: np.ndarray, model: YOLO, deepsort: DeepSort, stats: dict) -> np.ndarray:
    t0 = time.perf_counter()

    baseline_width, baseline_height = 1280, 960
    current_h, current_w = frame.shape[:2]
    scale_factor_w = current_w / baseline_width
    scale_factor_h = current_h / baseline_height
    scale_factor = (scale_factor_w + scale_factor_h) / 2.0

    # 2) YOLO 检测
    t_det0 = time.perf_counter()
    results = model(frame)
    t_det1 = time.perf_counter()
    boxes = results[0].boxes if len(results) > 0 else None

    detections = []
    confidences = []
    clss = []

    err_threshold = stats["err_threshold"]
    real_coin_diameter = stats["real_coin_diameter"]

    if boxes is not None:
        for box in boxes:
            x, y, w, h = box.xywh[0].detach().cpu().numpy()
            conf = float(box.conf[0].detach().cpu().numpy())
            cls = int(box.cls[0].detach().cpu().numpy())

            if conf < 0.3:
                continue

            # 硬币（cls == 0 示例），用于动态尺度校准
            if cls == 0:
                if abs(w - h) < err_threshold:
                    stats["h_scale"] = real_coin_diameter / h
                    stats["w_scale"] = real_coin_diameter / w

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    thickness_coin = max(1, int(2 * scale_factor))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness_coin)
                    info = f"h:{h * stats['h_scale']:.2f}mm w:{w * stats['w_scale']:.2f}mm"
                    cv2.putText(frame, info, (x1, max(0, y1 - int(5*scale_factor))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, (0, 0, 0), thickness_coin)
                    stats["err_threshold"] = abs(w - h)
                continue

            detections.append([x, y, w, h])  # 你的 DeepSORT 接口为 xywh
            confidences.append(conf)
            clss.append(cls)

    detections = np.array(detections) if len(detections) else np.empty((0, 4))
    confidences = np.array(confidences) if len(confidences) else np.empty((0,))
    clss = np.array(clss) if len(clss) else np.empty((0,))

    # 3) DeepSORT 跟踪
    t_trk0 = time.perf_counter()
    trackers = deepsort.update(detections, confidences, clss, frame)
    t_trk1 = time.perf_counter()

    # 4) 分类 + 可视化
    t_cls_sum = 0.0
    t_cls_cnt = 0

    for tracker in trackers:
        # x0, y0, x1, y1, cls_idx, track_id
        x0, y0, x1, y1, cls_idx, track_id = tracker[:6]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(current_w, x1), min(current_h, y1)

        cropped_img = frame[y0:y1, x0:x1]
        if cropped_img.size == 0:
            continue

        t_cls0 = time.perf_counter()
        # 二次分类：RiceLCNN
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        Img = Image.fromarray(cropped_img_rgb)

        new_image = Image.new("RGB", (224, 224), color=(50, 50, 50))
        w_c, h_c = Img.size
        aspect_ratio = w_c / float(h_c)
        new_w = int(224 * aspect_ratio)
        new_h = 224
        if new_w > 224:
            new_w = 224
            new_h = int(224 / aspect_ratio)

        paste_x = (224 - new_w) // 2
        paste_y = (224 - new_h) // 2
        resized_image = Img.resize((new_w, new_h), Image.LANCZOS)
        new_image.paste(resized_image, (paste_x, paste_y))

        new_image_arr = np.array(new_image)
        predicted_class, confidence_val = predict_image_from_numpy(new_image_arr)
        t_cls1 = time.perf_counter()
        t_cls_sum += (t_cls1 - t_cls0)
        t_cls_cnt += 1

        # 首次出现计数
        if track_id not in stats["seen_track_ids"]:
            stats["seen_track_ids"].add(track_id)
            stats["track_time"][track_id] = time.time()
            stats["track_id_count"][track_id] = stats["total_num"]
            stats["total_num"] += 1
            if predicted_class in stats["class_names"]:
                stats["class_names"][predicted_class] += 1

        elapsed_time = 0.0
        if track_id in stats["track_time"]:
            elapsed_time = time.time() - stats["track_time"][track_id]

        # 简单形态量测（基于 bbox 与动态尺度）
        box_h = abs(y1 - y0)
        box_w = abs(x1 - x0)
        major_axis = max(box_h, box_w)
        minor_axis = min(box_h, box_w)
        real_h = major_axis * stats["h_scale"]
        real_w = minor_axis * stats["w_scale"]

        # 绘制
        thickness = max(int(1 * scale_factor), 1)
        font_scale = 0.5 * scale_factor
        label = f"{predicted_class} ID:{stats['track_id_count'][track_id]} conf:{confidence_val:.2f}"
        info = f"h:{real_h:.2f}mm w:{real_w:.2f}mm  T:{elapsed_time:.1f}s"
        color = stats["color_map"].get(predicted_class, (0, 255, 0))

        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(frame, label, (x0, max(0, y0 - int(10 * scale_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, info, (x0, max(0, y0 - int(30 * scale_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # 5) 总体计数信息
    thickness_text = max(int(1 * scale_factor), 1)
    font_scale_text = 0.7 * scale_factor
    base_x = int(10 * scale_factor)
    base_y = int(50 * scale_factor)
    line_gap = int(20 * scale_factor)

    cv2.putText(frame, f"Total RiceSeed Num: {stats['total_num']}", (base_x, base_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (255, 0, 0), thickness_text)
    offset_y = base_y + int(30 * scale_factor)
    for idx, (cls_name, cls_cnt) in enumerate(stats["class_names"].items()):
        text_str = f"{cls_name}: {cls_cnt}"
        color_cls = stats["color_map"].get(cls_name, (0, 255, 0))
        cv2.putText(frame, text_str, (base_x, offset_y + int(idx * line_gap)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, color_cls, thickness_text)

    # 阶段耗时记录
    t1 = time.perf_counter()
    stats["timing"] = {
        "frame_ms": (t1 - t0) * 1000.0,
        "det_ms": (t_det1 - t_det0) * 1000.0,
        "trk_ms": (t_trk1 - t_trk0) * 1000.0,
        "cls_ms": (t_cls_sum / max(1, t_cls_cnt)) * 1000.0
    }
    return frame


# ===================== 主函数 =====================
def main():
    # A. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # B. 模型/跟踪器
    model_path = "runs/detect/train/weights/best.pt"   # 替换为你的 YOLOv11-LA 权重
    model = init_yolo(model_path, device)

    reid_weights = r"deep_sort/deep_sort/deep/checkpoint/ckpt.t7"  # 替换为你的 ReID 权重
    deepsort = init_deepsort(
        reid_weights, max_dist=0.2, min_confidence=0.3,
        nms_max_overlap=0.5, max_iou_distance=0.7, max_age=70,
        n_init=3, nn_budget=100, num_target=751
    )

    # C. 摄像头
    cap = init_webcam(camera_index=0)
    frame_rate, w, h = get_webcam_info(cap)
    print(f"[INFO] Camera: {w}x{h} @ ~{frame_rate:.1f} FPS")

    # D. 视频写入（Windows 推荐 mp4v）
    output_video_path = os.path.join("video", "001_1.mp4")
    vw = init_video_writer(output_video_path, frame_rate, (w, h))

    # E. 统计容器
    stats = {
        "real_coin_diameter": 100.0,     # mm，依据你的标定物直径修改
        "h_scale": 0.0,
        "w_scale": 0.0,
        "err_threshold": 999.0,
        "track_id_count": {},
        "seen_track_ids": set(),
        "track_time": {},
        "class_names": {
            "2022-1": 0, "2022-2": 0, "2022-3": 0, "2022-4": 0, "2022-5": 0,
            "2022-6": 0, "2022-7": 0, "2022-8": 0, "2022-9": 0
        },
        "total_num": 0,
        "color_map": None,
        "timing": {"frame_ms": 0.0, "det_ms": 0.0, "trk_ms": 0.0, "cls_ms": 0.0}
    }
    stats["color_map"] = generate_colors(stats["class_names"])

    # F. 系统监控器 + FPS EMA
    monitor = SystemMonitor()
    fps_ema = None
    ema_alpha = 0.1

    # G. 主循环
    cv2.namedWindow("Real-time Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Inference", min(w, 1280), min(h, 720))

    while True:
        loop_t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 无法读取摄像头画面或已结束。")
            break

        processed = process_frame(frame, model, deepsort, stats)

        loop_t1 = time.perf_counter()
        latency_ms = (loop_t1 - loop_t0) * 1000.0
        inst_fps = 1000.0 / max(1e-6, latency_ms)
        fps_ema = inst_fps if fps_ema is None else (1 - ema_alpha) * fps_ema + ema_alpha * inst_fps

        sysinfo = monitor.read()
        draw_hud(processed, fps_ema, latency_ms, sysinfo, origin=(10, 80))

        # 可选：阶段时延行
        t = stats.get("timing", {})
        stage_line = f"det:{t.get('det_ms',0):.1f}ms  trk:{t.get('trk_ms',0):.1f}ms  cls(avg):{t.get('cls_ms',0):.1f}ms"
        cv2.putText(processed, stage_line, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

        vw.write(processed)
        cv2.imshow("Real-time Inference", processed)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # H. 收尾
    cap.release()
    vw.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
