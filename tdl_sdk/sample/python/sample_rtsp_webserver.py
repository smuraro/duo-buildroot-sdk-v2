#!/usr/bin/env python3
"""
sample_rtsp_webserver.py — Câmera + inferência TDL + RTSP + interface web

Transmite o vídeo com overlay de detecções via RTSP (H264/H265) e serve
uma interface web que exibe o stream em tempo real via HTTP-FLV (mpegts.js).
O FFmpeg atua como proxy interno RTSP→FLV sem re-encoding.

Requer: ffmpeg disponível no PATH do dispositivo.

Uso:
    python3 sample_rtsp_webserver.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        [--model-type SCRFD_DET_FACE] \\
        [--width 1280] [--height 720] \\
        [--threshold 0.5] \\
        [--codec h264] [--bitrate 3072] [--gop 15] [--skip-every 1] [--persist-detections] \\
        [--session live] \\
        [--web-port 9000] \\
        [--labels "classe0,classe1"] \\
        [--mirror] [--flip]

Acesso:
    http://<ip-do-dispositivo>:9000/
Stream RTSP direto (VLC/ffplay):
    rtsp://<ip-do-dispositivo>:554/<session>
"""

import argparse
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """HTTP server que atende cada conexão numa thread separada.
    Necessário para servir /stream (bloqueante) em paralelo com /api/state.
    """
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return   # browser fechou a conexão — ignorar
        super().handle_error(request, client_address)

import tdl
from tdl import image, nn

# ─── COCO80 class names ───────────────────────────────────────────────────────

COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

# ─── Labels ───────────────────────────────────────────────────────────────────

_custom_labels: dict = {}


def _labels_from_factory(model_type_name: str) -> dict:
    try:
        types = nn.get_model_types(model_type_name)
        return {i: n for i, n in enumerate(types)} if types else {}
    except Exception:
        return {}


def _load_labels(labels_arg: str) -> dict:
    if not labels_arg:
        return {}
    if "," in labels_arg or not labels_arg.endswith(".txt"):
        names = [n.strip() for n in labels_arg.split(",")]
    else:
        try:
            with open(labels_arg) as f:
                names = [l.rstrip("\n") for l in f if l.strip()]
        except OSError as e:
            print(f"[AVISO] Não foi possível abrir --labels '{labels_arg}': {e}")
            return {}
    return {i: n for i, n in enumerate(names)}


def _resolve_name(class_id: int, class_name: str) -> str:
    if class_name.startswith("cls") or class_name == "UNDEFINED":
        if class_id in _custom_labels:
            return _custom_labels[class_id]
        if 0 <= class_id < len(COCO80_NAMES):
            return COCO80_NAMES[class_id]
    return class_name


def _draw_inference(frame, dets, is_keypoint: bool, threshold: float):
    """Dispatch inference result to the appropriate draw function.

    Handles all ModelOutputType variants:
      OBJECT_DETECTION / OBJECT_DETECTION_WITH_LANDMARKS / OBJECT_LANDMARKS
          → draw_detections or draw_keypoints
      CLASSIFICATION / CLS_ATTRIBUTE  → draw_classification
      SEGMENTATION                    → draw_segmentation
      OBJECT_DETECTION_WITH_SEGMENTATION → draw_instance_segmentation
      OCR_INFO                        → draw_ocr
      FEATURE_EMBEDDING (numpy array) → nothing (no visual representation)
    """
    import numpy as np
    if dets is None:
        return
    # FEATURE_EMBEDDING: numpy array — nothing to draw
    if isinstance(dets, np.ndarray):
        return
    if not dets:
        return

    if isinstance(dets, (list, tuple)):
        first = dets[0]
        # OCR_INFO: list containing a string
        if isinstance(first, str):
            image.draw_ocr(frame, dets)
            return
        if isinstance(first, dict):
            # SEGMENTATION: per-pixel class map
            if "output_width" in first:
                image.draw_segmentation(frame, dets)
                return
            # OBJECT_DETECTION_WITH_SEGMENTATION: bboxes + masks
            if "bboxes_seg" in first or "mask_width" in first:
                image.draw_instance_segmentation(frame, dets,
                                                 score_threshold=threshold)
                return
            # OBJECT_LANDMARKS: landmarks only, no bounding box
            if "landmarks" in first and "x1" not in first:
                image.draw_keypoints(frame, dets, score_threshold=threshold)
                return
            # CLASSIFICATION / CLS_ATTRIBUTE: no bbox, no landmarks
            if "x1" not in first and "landmarks" not in first and (
                    "class_id" in first or any(
                        k.endswith("_score") for k in first)):
                if "class_id" in first and "class_name" not in first:
                    dets = [{**d, "class_name": _resolve_name(
                                d.get("class_id", -1),
                                f"cls{d.get('class_id', 0)}")}
                            for d in dets if isinstance(d, dict)]
                image.draw_classification(frame, dets)
                return
            # OBJECT_DETECTION / OBJECT_DETECTION_WITH_LANDMARKS
            enriched = [{**d, "class_name": _resolve_name(
                            d.get("class_id", -1), d.get("class_name", ""))}
                        for d in dets]
            if "landmarks" in first:
                # Has both bbox and landmarks: draw box+label then landmark dots
                image.draw_detections(frame, enriched, score_threshold=threshold)
                image.draw_keypoints(frame, enriched, score_threshold=threshold)
            elif is_keypoint:
                image.draw_keypoints(frame, enriched, score_threshold=threshold)
            else:
                image.draw_detections(frame, enriched, score_threshold=threshold)
    elif isinstance(dets, dict):
        # Single dict result (some models return dict directly)
        if "output_width" in dets:
            image.draw_segmentation(frame, dets)
        elif "bboxes_seg" in dets:
            image.draw_instance_segmentation(frame, dets,
                                             score_threshold=threshold)
        elif "class_id" in dets or "is_male" in dets:
            image.draw_classification(frame, dets)


# ─── Shared state ─────────────────────────────────────────────────────────────

_state_lock         = threading.Lock()
_last_detections    = []
_det_lock           = threading.Lock()
_running            = True
_persist_detections = False   # configurado em main() via args.persist_detections
_last_detect_frame  = -1      # frame_idx da última inferência positiva

_infer_fps   = 0.0
_infer_ms    = 0.0
_cam_fps     = 0.0
_frame_count = 0
_det_total   = 0

# MJPEG fallback: latest JPEG bytes when ffmpeg is not available
_jpeg_lock   = threading.Lock()
_latest_jpeg = b""
_ffmpeg_available = False
_status     = "Aguardando modelo..."

# ─── Hot-swap detector ────────────────────────────────────────────────────────
# Protected by _detector_lock; swap happens atomically between frames.

_detector          = None        # current nn.Model instance
_detector_lock     = threading.Lock()
_current_model_type = ""         # model type name string (e.g. "SCRFD_DET_FACE")
_current_threshold  = 0.5
_switch_status      = ""         # last switch result message

_FACTORY_JSON_PATH = "/mnt/system/configs/model/model_factory.json"
_CVIMODEL_SEARCH_DIRS = ["/root", "/root/cv181x", "/mnt", "/mnt/data"]

# ─── Signal handling ──────────────────────────────────────────────────────────


def _sigint_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)

# ─── Inference / release thread ───────────────────────────────────────────────
#
# Todos os frames lidos passam pela _release_queue antes de serem devolvidos
# à câmera.  A thread processa-os em ordem FIFO: faz inferência nos marcados
# com do_infer=True, depois chama sempre cam.release().
#
# Por que isso é necessário:
#   ViDecoder::release() usa uma fila interna (frameQueues) e devolve sempre o
#   frame mais antigo.  Se camera_loop chamasse cam.release() para o frame N
#   enquanto _inference_worker ainda usa o frame N-1 ou N, a liberação cairia
#   sobre o VB block errado → "vb released" no VPSS do preprocessador.
#
# Regra: cam.release() só é chamado dentro desta thread, garantindo ordem FIFO.

_release_queue       = []
_release_lock        = threading.Lock()
_MAX_RELEASE_PENDING = 2   # máx. frames aguardando além do que está em inferência

# Pool de VB blocks da câmera — ver comentário análogo em sample_rtsp_server.py.
_VB_BUFFER_NUM = 5   # pool size passado ao Camera(); semáforo = _VB_BUFFER_NUM - 2


def _inference_worker(cam, vb_sem):
    global _running, _infer_fps, _infer_ms, _last_detect_frame
    t0    = time.time()
    count = 0
    while _running:
        item = None
        with _release_lock:
            if _release_queue:
                item = _release_queue.pop(0)
        if item is None:
            time.sleep(0.001)
            continue

        frame, do_infer = item
        if do_infer:
            # Mantém _detector_lock durante toda a inferência.
            # _do_switch_model também usa _detector_lock para trocar o modelo e
            # depois chama old.close().  Como old.close() fica FORA do lock,
            # ele só é executado depois que este bloco terminar — garantindo que
            # o NPU runtime nunca é liberado enquanto inference() o está a usar.
            with _detector_lock:
                det = _detector
                if det is not None:
                    try:
                        ti   = time.time()
                        dets = det.inference(frame)
                        _infer_ms = (time.time() - ti) * 1000
                        with _det_lock:
                            _last_detections[:] = dets
                            if dets:
                                _last_detect_frame = _frame_count
                    except Exception as e:
                        print(f"[AVISO] inference error: {e}")

                    count += 1
                    elapsed = time.time() - t0
                    if elapsed > 0:
                        _infer_fps = count / elapsed

        # Sempre libera em ordem FIFO — nunca camera_loop chama cam.release()
        cam.release()
        vb_sem.release()   # libera um slot do pool para o main thread


# ─── Model hot-swap helpers ───────────────────────────────────────────────────

_model_types_cache: list = []

def _list_model_types() -> list:
    """Return all model type names available in model_factory.json via TDL."""
    return _model_types_cache


_PLATFORM = "cv181x"   # platform suffix used in .cvimodel filenames

def _cvimodels_for_type(model_type_name: str) -> list:
    """Return the cvimodel path(s) associated with a model type.

    Uses model_factory.json's file_name field to build the expected path and
    checks each search dir for an existing file.  Falls back to a filesystem
    scan only when the factory returns no file_name.
    """
    try:
        base = nn.get_model_filename(model_type_name)
    except Exception:
        base = ""

    found = []
    if base:
        # Construct expected filenames: base_platform.cvimodel (standard)
        # and base.cvimodel (when the file already carries the extension).
        candidates = [
            f"{base}_{_PLATFORM}.cvimodel",
            f"{base}.cvimodel",
        ]
        for d in _CVIMODEL_SEARCH_DIRS:
            platform_dir = os.path.join(d, _PLATFORM)
            for c in candidates:
                for search_root in (platform_dir, d):
                    full = os.path.join(search_root, c)
                    if os.path.isfile(full) and full not in found:
                        found.append(full)
    if not found:
        # Fallback: scan search dirs for any .cvimodel (slow, first call only)
        for d in _CVIMODEL_SEARCH_DIRS:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".cvimodel"):
                        full = os.path.join(root, f)
                        if full not in found:
                            found.append(full)
    return sorted(found)


def _do_switch_model(model_type_name: str, model_path: str) -> tuple:
    """Load a new model and atomically replace the running detector.

    Returns (ok: bool, message: str).
    """
    global _detector, _current_model_type, _current_threshold
    global _custom_labels, _switch_status

    model_type = getattr(nn.ModelType, model_type_name, None)
    if model_type is None:
        return False, f"ModelType desconhecido: {model_type_name}"
    if not os.path.isfile(model_path):
        return False, f"Arquivo não encontrado: {model_path}"

    try:
        new_det = nn.get_model(model_type, model_path)
        new_det.set_threshold(_current_threshold)
    except Exception as e:
        return False, f"Erro ao carregar modelo: {e}"

    # Auto-load labels from factory for the new model type
    new_labels = _load_labels("") or _labels_from_factory(model_type_name)

    # Atomic swap — inference worker will pick up new detector on next frame
    with _detector_lock:
        old = _detector
        _detector           = new_det
        _current_model_type = model_type_name
        _custom_labels      = new_labels
    global _status
    _status = "Running"

    # Release old detector VPSS group after swap
    if old is not None:
        try:
            old.close()
        except Exception:
            pass

    # Clear stale detections from previous model
    with _det_lock:
        _last_detections.clear()

    Handler.model_name = os.path.basename(model_path)
    msg = f"Modelo trocado: {model_type_name} / {os.path.basename(model_path)}"
    _switch_status = msg
    print(f"[switch] {msg}")
    return True, msg


# ─── Main camera + RTSP loop ──────────────────────────────────────────────────


def camera_loop(args, rtsp):
    global _running, _cam_fps, _frame_count, _det_total, _status

    with _detector_lock:
        _status = "Running" if _detector is not None else "Aguardando modelo"
    limit      = args.frames if args.frames > 0 else None
    frame_idx  = 0
    t_start    = time.time()
    t_report   = t_start
    skip_every = args.skip_every

    _cam_ms_acc  = 0.0
    _draw_ms_acc = 0.0
    _send_ms_acc = 0.0
    _acc_count   = 0

    cam = image.Camera(args.width, args.height, image.ImageFormat.YUV420SP_VU,
                       vb_buffer_num=_VB_BUFFER_NUM,
                       mirror=args.mirror, flip=args.flip)

    # Semáforo limita frames em user space a (_VB_BUFFER_NUM - 2), evitando
    # esgotamento do pool de VB blocks e stall do ISP.
    vb_sem = threading.Semaphore(_VB_BUFFER_NUM - 2)

    # Thread de inferência/release iniciada aqui, depois de cam ser criada,
    # para que possa chamar cam.release() em ordem FIFO.
    infer_thread = threading.Thread(
        target=_inference_worker, args=(cam, vb_sem), daemon=True)
    infer_thread.start()

    try:
        while _running and (limit is None or frame_idx < limit):
            # Aguarda um slot livre antes de ler o próximo frame.
            if not vb_sem.acquire(timeout=0.5):
                continue   # timeout — verifica _running e tenta de novo
            t0    = time.time()
            frame = cam.read()
            cam_ms = (time.time() - t0) * 1000

            with _det_lock:
                dets = list(_last_detections)
                det_frame = _last_detect_frame

            # Com persist_detections: aplica TTL de skip_every frames.
            # Se a última inferência positiva foi há >= skip_every frames, expira.
            if _persist_detections and (frame_idx - det_frame) >= skip_every:
                dets = []

            t1 = time.time()
            if dets:
                _det_total += len(dets) if isinstance(dets, (list, tuple)) else 1
                mt = _current_model_type.upper()
                is_kp = "KEYPOINT" in mt or "POSE" in mt
                _draw_inference(frame, dets, is_kp, _current_threshold)
            draw_ms = (time.time() - t1) * 1000

            t2 = time.time()
            rtsp.send_frame(frame)
            send_ms = (time.time() - t2) * 1000

            # MJPEG fallback: encode JPEG para browsers sem ffmpeg
            if not _ffmpeg_available:
                jpeg = image.frame_to_jpeg(frame, quality=50, scale=1.0)
                with _jpeg_lock:
                    global _latest_jpeg
                    _latest_jpeg = jpeg

            # Enfileira APÓS rtsp.send_frame e frame_to_jpeg: o main thread
            # já terminou de usar o VB block; a thread de inferência pode
            # agora acessá-lo e depois liberá-lo em ordem FIFO via cam.release().
            do_infer = (frame_idx % skip_every == 0)
            with _release_lock:
                if len(_release_queue) >= _MAX_RELEASE_PENDING:
                    do_infer = False
                _release_queue.append((frame, do_infer))

            frame_idx += 1

            _cam_ms_acc  += cam_ms
            _draw_ms_acc += draw_ms
            _send_ms_acc += send_ms
            _acc_count   += 1

            now = time.time()
            elapsed = now - t_start
            _cam_fps     = frame_idx / elapsed if elapsed > 0 else 0
            _frame_count = frame_idx

            if now - t_report >= 5.0:
                n = max(_acc_count, 1)
                print(f"  frame {frame_idx:6d}  "
                      f"cam={_cam_fps:5.1f}fps  "
                      f"inf={_infer_fps:4.1f}fps  "
                      f"dets={len(dets)}  "
                      f"| read={_cam_ms_acc/n:5.1f}ms  "
                      f"inf={_infer_ms:5.1f}ms  "
                      f"draw={_draw_ms_acc/n:4.1f}ms  "
                      f"send={_send_ms_acc/n:5.1f}ms")
                _cam_ms_acc = _draw_ms_acc = _send_ms_acc = 0.0
                _acc_count  = 0
                t_report    = now

    except Exception as exc:
        _status = f"Erro: {exc}"
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        vb_sem.release()   # desbloqueia acquire() caso esteja esperando
        infer_thread.join(timeout=2.0)
        # Frames pendentes na fila são descartados; cam.close() libera o
        # frameQueues interno do ViDecoder (todos os VB blocks restantes).
        with _release_lock:
            _release_queue.clear()
        cam.close()


# ─── HTML page ────────────────────────────────────────────────────────────────

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vision Inference — RTSP</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;600;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0c0f;
    --panel: #111318;
    --border: #1e2330;
    --accent: #00e5ff;
    --accent2: #39ff14;
    --warn: #ff4f1f;
    --text: #c8d0e0;
    --dim: #4a5268;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Barlow', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px);
    pointer-events: none;
    z-index: 1000;
  }

  header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }

  .logo-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
    animation: blink 1.4s ease-in-out infinite;
  }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  header h1 {
    font-size: 13px; font-weight: 600;
    letter-spacing: .25em; text-transform: uppercase;
    color: var(--accent);
  }

  header .sub { font-family: var(--mono); font-size: 11px; color: var(--dim); margin-left: auto; }

  main {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 0;
    flex: 1;
  }

  .feed-wrap {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid var(--border);
    min-height: 420px;
  }

  .feed-wrap video {
    max-width: 100%;
    max-height: calc(100vh - 120px);
    display: block;
    object-fit: contain;
    width: 100%;
  }

  .no-feed {
    position: absolute;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 13px;
  }

  .spinner {
    width: 36px; height: 36px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .9s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .corner {
    position: absolute;
    width: 22px; height: 22px;
    border-color: var(--accent);
    border-style: solid;
    opacity: .6;
  }
  .corner.tl { top:12px; left:12px;   border-width:2px 0 0 2px; }
  .corner.tr { top:12px; right:12px;  border-width:2px 2px 0 0; }
  .corner.bl { bottom:12px; left:12px;  border-width:0 0 2px 2px; }
  .corner.br { bottom:12px; right:12px; border-width:0 2px 2px 0; }

  .panel {
    display: flex; flex-direction: column;
    background: var(--panel);
    overflow-y: auto;
  }

  .section { padding: 20px; border-bottom: 1px solid var(--border); }

  .section-title {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: .2em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 14px;
  }

  .status-badge {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--mono); font-size: 12px; color: var(--accent2);
  }
  .status-badge.warn { color: var(--warn); }
  .status-dot {
    width:7px; height:7px; border-radius:50%;
    background:currentColor; box-shadow:0 0 6px currentColor;
    animation: blink 1.4s ease-in-out infinite;
  }

  .metric-row {
    display:flex; justify-content:space-between; align-items:baseline;
    padding:6px 0; border-bottom:1px solid var(--border); font-size:13px;
  }
  .metric-row:last-child { border-bottom:none; }
  .metric-label { color:var(--dim); font-family:var(--mono); font-size:11px; }
  .metric-val   { font-family:var(--mono); font-weight:600; color:var(--text); }
  .metric-val.accent { color:var(--accent); }

  .perf-bar-wrap { margin-top:4px; }
  .perf-bar-label {
    display:flex; justify-content:space-between;
    font-family:var(--mono); font-size:10px; color:var(--dim); margin-bottom:3px;
  }
  .perf-bar-bg {
    height:4px; background:var(--border); border-radius:2px;
    margin-bottom:8px; overflow:hidden;
  }
  .perf-bar-fill {
    height:100%; border-radius:2px; background:var(--accent);
    transition:width .3s ease;
  }
  .perf-bar-fill.draw  { background:var(--accent2); }
  .perf-bar-fill.send  { background:#ff9f1c; }

  .det-list { list-style:none; }
  .det-item {
    display:flex; align-items:center; gap:8px;
    padding:7px 0; border-bottom:1px solid var(--border);
    font-family:var(--mono); font-size:12px; color:var(--text);
  }
  .det-item:last-child { border-bottom:none; }
  .det-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
  .no-det  { font-family:var(--mono); font-size:12px; color:var(--dim); }

  .rtsp-url {
    font-family:var(--mono); font-size:10px; color:var(--dim);
    word-break:break-all; padding:6px 0;
  }
  .rtsp-url a { color:var(--accent); text-decoration:none; }

  footer {
    padding:10px 28px; border-top:1px solid var(--border);
    font-family:var(--mono); font-size:10px; color:var(--dim);
    display:flex; gap:24px; background:var(--panel);
  }

  @media (max-width:700px) {
    main { grid-template-columns:1fr; }
    .feed-wrap { border-right:none; border-bottom:1px solid var(--border); }
    .panel { max-height:340px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>Vision Inference — RTSP</h1>
  <span class="sub" id="hdr-model">--</span>
</header>

<main>
  <div class="feed-wrap">
    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>
    <div class="no-feed" id="no-feed">
      <div class="spinner"></div>
      <span id="status-text">Aguardando stream...</span>
    </div>
    <video id="live-video" autoplay muted playsinline style="display:none"></video>
    <img id="live-mjpeg" style="display:none;max-width:100%;max-height:calc(100vh - 120px);object-fit:contain" alt="MJPEG stream">
  </div>

  <aside class="panel">
    <div class="section">
      <div class="section-title">System</div>
      <div class="metric-row">
        <span class="metric-label">STATUS</span>
        <span class="status-badge" id="status-badge">
          <span class="status-dot"></span>
          <span id="badge-text">init</span>
        </span>
      </div>
      <div class="metric-row">
        <span class="metric-label">FRAMES</span>
        <span class="metric-val accent" id="frame-count">0</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">CAM FPS</span>
        <span class="metric-val" id="cam-fps">--</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">INF FPS</span>
        <span class="metric-val" id="inf-fps">--</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Timing (ms)</div>
      <div class="perf-bar-wrap">
        <div class="perf-bar-label"><span>INF</span><span id="inf-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill" id="inf-bar" style="width:0%"></div></div>
        <div class="perf-bar-label"><span>DRAW</span><span id="draw-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill draw" id="draw-bar" style="width:0%"></div></div>
        <div class="perf-bar-label"><span>SEND</span><span id="send-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill send" id="send-bar" style="width:0%"></div></div>
      </div>
    </div>

    <div class="section" style="flex:1">
      <div class="section-title">Detections</div>
      <ul class="det-list" id="det-list">
        <li class="no-det">No detections yet</li>
      </ul>
    </div>

    <div class="section">
      <div class="section-title">RTSP direto</div>
      <div class="rtsp-url" id="rtsp-url">--</div>
    </div>

    <div class="section">
      <div class="section-title">Trocar modelo</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <select id="sel-type" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:5px 6px;font-family:var(--mono);font-size:11px;border-radius:3px">
          <option value="">-- model type --</option>
        </select>
        <select id="sel-file" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:5px 6px;font-family:var(--mono);font-size:11px;border-radius:3px">
          <option value="">-- cvimodel --</option>
        </select>
        <button id="btn-switch" onclick="doSwitch()"
          style="background:var(--accent);color:#000;border:none;padding:7px;font-family:var(--mono);font-size:11px;font-weight:700;border-radius:3px;cursor:pointer;letter-spacing:.1em">
          APLICAR
        </button>
        <div id="switch-msg" style="font-family:var(--mono);font-size:10px;color:var(--dim);min-height:14px"></div>
      </div>
    </div>
  </aside>
</main>

<footer>
  <span>TDL RTSP</span>
  <span id="footer-model">model: --</span>
  <span id="footer-time">--</span>
</footer>

<!-- mpegts.js — HTTP-FLV player (https://github.com/xqq/mpegts.js) -->
<script src="https://cdn.jsdelivr.net/npm/mpegts.js/dist/mpegts.min.js"
        onerror="onMpegtsLoadError()"></script>
<script>
const COLORS = [
  '#00e5ff','#39ff14','#ff4f1f','#0080ff',
  '#ff00ff','#ffff00','#ff8000','#8000ff'
];

function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }
function barWidth(ms, maxMs=500){ return clamp(ms/maxMs*100,0,100).toFixed(1)+'%'; }

// ─── Player (FLV via mpegts.js  ou  MJPEG nativo) ───────────────────────────

let player      = null;
let streamMode  = null;   // 'flv' | 'mjpeg' — definido pela 1ª resposta /api/state
let _stallTimer = null;
let _lastTime   = -1;

function clickToPlay() {
  const video = document.getElementById('live-video');
  const nf    = document.getElementById('no-feed');
  video.play().then(() => { nf.style.display = 'none'; }).catch(() => {});
}

function setFeedStatus(msg) {
  const el = document.getElementById('status-text');
  if (el) el.textContent = msg;
}

function _destroyPlayer() {
  if (_stallTimer) { clearInterval(_stallTimer); _stallTimer = null; }
  if (player) { player.destroy(); player = null; }
  _lastTime = -1;
}

function _reconnect(reason) {
  setFeedStatus('Reconectando' + (reason ? ': ' + reason : '...'));
  const nf = document.getElementById('no-feed');
  nf.style.display = 'flex';
  _destroyPlayer();
  setTimeout(startFlvPlayer, 3000);
}

function _startStallWatchdog(video) {
  _lastTime = video.currentTime;
  _stallTimer = setInterval(() => {
    if (!player) return;
    if (video.paused || video.ended) return;
    if (video.currentTime === _lastTime) {
      _reconnect('stall');
    } else {
      _lastTime = video.currentTime;
    }
  }, 6000);
}

function startFlvPlayer() {
  if (!window.mpegts) {
    showFeedError('mpegts.js nao carregou (CDN inacessivel).');
    return;
  }
  if (!mpegts.isSupported()) {
    showFeedError('Browser nao suporta MSE — use VLC: ' + (document.getElementById('rtsp-url') || {}).textContent);
    return;
  }
  if (player) return;

  setFeedStatus('Conectando ao stream...');
  const video = document.getElementById('live-video');
  player = mpegts.createPlayer(
    { type: 'flv', url: '/stream', isLive: true },
    {
      enableWorker: false,
      liveBufferLatencyChasing: true,
      liveBufferLatencyMaxLatency: 2.0,
      liveBufferLatencyMinRemain: 0.5,
    }
  );
  player.attachMediaElement(video);
  // Startup watchdog: if MEDIA_INFO doesn't arrive in 10 s, reconnect
  const _thisPlayer = player;
  setTimeout(() => {
    if (player === _thisPlayer && document.getElementById('live-video').style.display === 'none') {
      _reconnect('timeout');
    }
  }, 10000);

  player.on(mpegts.Events.MEDIA_INFO, () => {
    showFeed('video');
    video.play().catch(() => {
      const nf = document.getElementById('no-feed');
      nf.style.display = 'flex';
      nf.innerHTML = '<span style="color:var(--accent);font-family:var(--mono);font-size:14px;cursor:pointer;padding:12px" onclick="clickToPlay()">&#9654; Clique para iniciar</span>';
    });
    _startStallWatchdog(video);
  });
  player.on(mpegts.Events.LOADING_COMPLETE, () => { _reconnect('ended'); });
  player.on(mpegts.Events.ERROR, (errType) => { _reconnect(errType); });
  player.load();
}

function startMjpegPlayer() {
  const img = document.getElementById('live-mjpeg');
  img.src = '/stream?' + Date.now();
  img.onload  = () => showFeed('mjpeg');
  img.onerror = () => { setFeedStatus('Reconectando...'); setTimeout(startMjpegPlayer, 3000); };
}

function showFeed(type) {
  document.getElementById('no-feed').style.display   = 'none';
  document.getElementById('live-video').style.display = type === 'video' ? 'block' : 'none';
  document.getElementById('live-mjpeg').style.display = type === 'mjpeg' ? 'block' : 'none';
}

function showFeedError(msg) {
  document.getElementById('no-feed').innerHTML =
    `<span style="color:var(--warn);font-family:var(--mono);font-size:12px;text-align:center">${msg}</span>`;
}

function onMpegtsLoadError() {
  // CDN inacessível — marcar para que initPlayer use MJPEG se disponível, ou mostre erro
  window.mpegts = null;
}

function initPlayer(mode) {
  if (streamMode === mode) return;
  streamMode = mode;
  if (mode === 'flv')   startFlvPlayer();
  if (mode === 'mjpeg') startMjpegPlayer();
}

// ─── Stats polling ───────────────────────────────────────────────────────────

async function poll() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const d = await r.json();

    document.getElementById('hdr-model').textContent =
      (d.model_type ? d.model_type + '  ' : '') + (d.model || '--');

    if (d.stream_mode) initPlayer(d.stream_mode);

    const badge     = document.getElementById('status-badge');
    const badgeText = document.getElementById('badge-text');
    const running   = d.status === 'Running';
    badge.className = 'status-badge' + (running ? '' : ' warn');
    badgeText.textContent = d.status || 'init';

    document.getElementById('frame-count').textContent = d.frame_count || 0;
    document.getElementById('cam-fps').textContent = (d.cam_fps||0).toFixed(1);
    document.getElementById('inf-fps').textContent = (d.inf_fps||0).toFixed(1);

    const perf = d.perf || [0, 0, 0];
    document.getElementById('inf-val').textContent  = perf[0];
    document.getElementById('draw-val').textContent = perf[1];
    document.getElementById('send-val').textContent = perf[2];
    document.getElementById('inf-bar').style.width  = barWidth(perf[0]);
    document.getElementById('draw-bar').style.width = barWidth(perf[1]);
    document.getElementById('send-bar').style.width = barWidth(perf[2]);

    const dets = d.detections || [];
    const list  = document.getElementById('det-list');
    if (dets.length === 0) {
      list.innerHTML = '<li class="no-det">No detections</li>';
    } else {
      list.innerHTML = dets.map((lbl, i) =>
        `<li class="det-item">
           <span class="det-dot" style="background:${COLORS[i%COLORS.length]}"></span>
           <span>${lbl}</span>
         </li>`
      ).join('');
    }

    if (d.rtsp_url) {
      document.getElementById('rtsp-url').innerHTML =
        `<a href="#" onclick="return false">${d.rtsp_url}</a>`;
    }

    document.getElementById('footer-model').textContent = 'model: ' + (d.model||'--');
    document.getElementById('footer-time').textContent  = new Date().toLocaleTimeString();

  } catch(e) { /* ignore */ }
}

setInterval(poll, 300);
poll();

// ─── Model selector ──────────────────────────────────────────────────────────

async function loadModelSelector() {
  try {
    const types = await fetch('/api/models').then(r => r.json());
    const selType = document.getElementById('sel-type');
    selType.innerHTML = '<option value="">-- model type --</option>';
    types.forEach(t => {
      const o = document.createElement('option');
      o.value = o.textContent = t;
      selType.appendChild(o);
    });
    selType.onchange = () => onTypeChange(selType.value);
  } catch(e) {
    document.getElementById('switch-msg').textContent = 'Erro ao carregar tipos';
  }
}

async function onTypeChange(modelType) {
  const selFile = document.getElementById('sel-file');
  const msg     = document.getElementById('switch-msg');
  if (!modelType) {
    selFile.innerHTML = '<option value="">-- cvimodel --</option>';
    return;
  }
  selFile.innerHTML = '<option value="">Buscando...</option>';
  msg.textContent   = '';
  try {
    const files = await fetch('/api/cvimodels?type=' + encodeURIComponent(modelType))
                    .then(r => r.json());
    selFile.innerHTML = '<option value="">-- cvimodel --</option>';
    if (files.length === 0) {
      selFile.innerHTML += '<option value="" disabled>Nenhum arquivo encontrado</option>';
    } else {
      files.forEach(f => {
        const o = document.createElement('option');
        o.value = f;
        o.textContent = f.split('/').pop();
        o.title = f;
        selFile.appendChild(o);
      });
      if (files.length === 1) selFile.value = files[0];
    }
  } catch(e) {
    selFile.innerHTML = '<option value="">Erro ao buscar arquivo</option>';
  }
}

async function doSwitch() {
  const modelType = document.getElementById('sel-type').value;
  const modelPath = document.getElementById('sel-file').value;
  const msg       = document.getElementById('switch-msg');
  const btn       = document.getElementById('btn-switch');

  if (!modelType || !modelPath) {
    msg.style.color = 'var(--warn)';
    msg.textContent = 'Selecione tipo e arquivo';
    return;
  }

  btn.disabled    = true;
  btn.textContent = 'Aguarde...';
  msg.style.color = 'var(--dim)';
  msg.textContent = 'Carregando...';

  try {
    const r = await fetch('/api/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_type: modelType, model_path: modelPath}),
    });
    const d = await r.json();
    msg.style.color = d.ok ? 'var(--accent2)' : 'var(--warn)';
    msg.textContent = d.message;
  } catch(e) {
    msg.style.color = 'var(--warn)';
    msg.textContent = 'Erro de comunicação';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'APLICAR';
  }
}

loadModelSelector();
</script>
</body>
</html>
"""

# ─── HTTP Handler ─────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    model_name   = ""
    session_name = "live"
    rtsp_url     = ""

    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            if _ffmpeg_available:
                self._proxy_flv()
            else:
                self._serve_mjpeg()

        elif self.path == "/api/state":
            import numpy as np
            with _det_lock:
                dets = _last_detections
                dets = list(dets) if isinstance(dets, (list, tuple)) else []
            labels = []
            for d in dets:
                if not isinstance(d, dict):
                    continue
                # Check if any *_score attribute key is present → CLS_ATTRIBUTE
                attr_scores = {k[:-6]: v for k, v in d.items()
                               if k.endswith("_score") and isinstance(v, float)}
                if attr_scores:
                    _ATTR_LABEL = {
                        "gender":   lambda d, s: ("Male" if d.get("is_male") else "Female") + f" {s:.0%}",
                        "age":      lambda d, s: f"Age {d['age']}" if "age" in d else f"Age {s*100:.0f}",
                        "glasses":  lambda d, s: ("Glasses" if d.get("is_wearing_glasses") else "No glasses") + f" {s:.0%}",
                        "mask":     lambda d, s: ("Mask" if d.get("is_wearing_mask") else "No mask") + f" {s:.0%}",
                        "hat":      lambda d, s: ("Hat" if d.get("is_wearing_hat") else "No hat") + f" {s:.0%}",
                        "emotion":  lambda d, s: f"Emotion {s:.2f}",
                        "pose":     lambda d, s: f"Pose {s:.2f}",
                        "blurness": lambda d, s: f"Blur {s:.2f}",
                    }
                    _ORDER = ["gender", "age", "glasses", "mask", "hat",
                              "emotion", "pose", "blurness"]
                    parts = []
                    for name in _ORDER:
                        if name not in attr_scores:
                            continue
                        fn = _ATTR_LABEL.get(name)
                        parts.append(fn(d, attr_scores[name]) if fn else f"{name} {attr_scores[name]:.2f}")
                    # Unknown attr_* keys
                    for name, sc in attr_scores.items():
                        if name not in _ORDER:
                            parts.append(f"{name} {sc:.2f}")
                    labels.append("  |  ".join(parts))
                else:
                    cls   = _resolve_name(d.get("class_id", -1), d.get("class_name", "?"))
                    score = d.get("score", 0.0)
                    labels.append(f"{cls} {score:.0%}")

            payload = json.dumps({
                "status":       _status,
                "frame_count":  _frame_count,
                "cam_fps":      round(_cam_fps,  1),
                "inf_fps":      round(_infer_fps, 1),
                "perf":         [int(_infer_ms), 0, 0],
                "detections":   labels,
                "model":        Handler.model_name,
                "model_type":   _current_model_type,
                "rtsp_url":     Handler.rtsp_url,
                "stream_mode":  "flv" if _ffmpeg_available else "mjpeg",
                "switch_status": _switch_status,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/api/models":
            payload = json.dumps(_list_model_types()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path.startswith("/api/cvimodels"):
            from urllib.parse import urlparse, parse_qs
            qs  = parse_qs(urlparse(self.path).query)
            typ = qs.get("type", [""])[0]
            payload = json.dumps(_cvimodels_for_type(typ) if typ else []).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/switch":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                req  = json.loads(body)
                ok, msg = _do_switch_model(req.get("model_type", ""),
                                           req.get("model_path", ""))
            except Exception as e:
                ok, msg = False, str(e)
            payload = json.dumps({"ok": ok, "message": msg}).encode()
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_mjpeg(self):
        """Fallback MJPEG stream quando ffmpeg não está disponível.
        Browsers suportam nativamente via <img src="/stream">.
        """
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace;boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while _running:
                with _jpeg_lock:
                    jpeg = _latest_jpeg
                if not jpeg:
                    time.sleep(0.05)
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                )
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
                time.sleep(0.04)   # ~25 fps máximo
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _proxy_flv(self):
        """Pull from the local RTSP server and push HTTP-FLV to the browser.

        FFmpeg remuxes the H264/H265 stream into an FLV container without
        re-encoding. The browser uses mpegts.js to decode it.
        """
        cmd = [
            "ffmpeg",
            "-loglevel",       "error",
            "-rtsp_transport", "tcp",
            "-i",              f"rtsp://127.0.0.1:554/{Handler.session_name}",
            "-c:v",            "copy",
            "-an",
            "-f",              "flv",
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            body = b"ffmpeg nao encontrado no PATH do dispositivo"
            self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        # Aguardar primeiros bytes com timeout de 10 s
        import select
        ready, _, _ = select.select([proc.stdout], [], [], 10.0)
        first_chunk = proc.stdout.read(4096) if ready else b""
        if not first_chunk:
            proc.kill()
            proc.wait()
            err = proc.stderr.read(512).decode(errors="replace").strip()
            body = f"Falha FFmpeg: {err or 'timeout / sem dados do RTSP'}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "video/x-flv")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self.wfile.write(first_chunk)
            self.wfile.flush()
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.kill()
            proc.wait()


# ─── Arguments ────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Câmera + inferência TDL + RTSP + interface web (HTTP-FLV)")
    p.add_argument("--model", default="",
                   help="Caminho para o arquivo .cvimodel (opcional; pode ser "
                        "selecionado na interface web)")
    p.add_argument("--model-type", default="",
                   help="Nome do ModelType (opcional; pode ser selecionado na "
                        "interface web)")
    p.add_argument("--width",     type=int, default=1280)
    p.add_argument("--height",    type=int, default=720)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--codec",     default="h264", choices=["h264", "h265"])
    p.add_argument("--bitrate",   type=int, default=3072,
                   help="Bitrate de codificação em kbps (padrão: 3072). "
                        "Aumente para melhor qualidade em movimento (ex: 4096, 6144).")
    p.add_argument("--gop",       type=int, default=15,
                   help="Intervalo de keyframe em frames (padrão: 15). "
                        "Menor = melhor qualidade em movimento; maior = melhor compressão estática.")
    p.add_argument("--session",   default="live",
                   help="Nome da sessão RTSP (padrão: live)")
    p.add_argument("--frames",     type=int, default=0,
                   help="Número de frames a transmitir; 0 = infinito (padrão: 0)")
    p.add_argument("--skip-every", type=int, default=1, dest="skip_every",
                   help="Executa inferência em 1 de cada N frames (padrão: 1). "
                        "Ex: 2 = inferência a cada 2 frames, reduz carga da CPU.")
    p.add_argument("--persist-detections", action="store_true", default=False,
                   dest="persist_detections",
                   help="Mantém a última detecção na tela enquanto não houver nova "
                        "detecção positiva (reduz flickering com --skip-every > 1).")
    p.add_argument("--web-port",   type=int, default=9000, dest="web_port")
    p.add_argument("--mirror",    action="store_true", default=False,
                   help="Espelhar horizontalmente (flip esquerda↔direita).")
    p.add_argument("--flip",      action="store_true", default=False,
                   help="Inverter verticalmente (flip cima↔baixo).")
    p.add_argument("--labels",    default="", dest="labels",
                   help="Nomes das classes: arquivo .txt ou 'cls0,cls1,...'")
    return p.parse_args()


# ─── Entry point ──────────────────────────────────────────────────────────────


def main():
    global _custom_labels, _status
    global _detector, _current_model_type, _current_threshold

    args = parse_args()
    _current_threshold = args.threshold

    # Cache model type list (loads model_factory.json — fast, no model needed)
    global _model_types_cache
    try:
        _model_types_cache = sorted(nn.get_available_model_types())
        print(f"Model types disponíveis: {len(_model_types_cache)}")
    except Exception as e:
        print(f"[AVISO] get_available_model_types: {e}")

    # Modelo inicial (opcional)
    if args.model and args.model_type:
        if args.labels:
            _custom_labels = _load_labels(args.labels)
            print(f"Labels carregados  : {len(_custom_labels)} classes (--labels)")
        else:
            _custom_labels = _labels_from_factory(args.model_type)
            if _custom_labels:
                print(f"Labels carregados  : {len(_custom_labels)} classes (model_factory.json)")

        model_type = getattr(nn.ModelType, args.model_type, None)
        if model_type is None:
            print(f"[ERRO] ModelType desconhecido: {args.model_type}")
            sys.exit(1)

        print(f"Carregando modelo  : {args.model}")
        detector = nn.get_model(model_type, args.model)
        detector.set_threshold(args.threshold)
        print(f"Limiar             : {detector.get_threshold():.2f}")

        with _detector_lock:
            _detector           = detector
            _current_model_type = args.model_type
    elif args.model or args.model_type:
        print("[AVISO] Forneça --model e --model-type juntos, ou nenhum dos dois.")
        print("        Iniciando sem modelo — selecione na interface web.")
    else:
        print("Nenhum modelo inicial — selecione na interface web.")

    # RTSP server
    print(f"\nIniciando servidor RTSP {args.width}x{args.height} "
          f"codec={args.codec} bitrate={args.bitrate}kbps gop={args.gop} sessão={args.session} ...")
    rtsp = image.RTSPServer(args.width, args.height, chn=0,
                            codec=args.codec, session_name=args.session,
                            bitrate=args.bitrate, gop=args.gop)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        device_ip = s.getsockname()[0]
        s.close()
    except Exception:
        device_ip = "127.0.0.1"
    rtsp_url = f"rtsp://{device_ip}:554/{args.session}"
    print(f"  Stream RTSP : {rtsp_url}")

    # HTTP handler config
    Handler.model_name   = os.path.basename(args.model) if args.model else ""
    Handler.session_name = args.session
    Handler.rtsp_url     = rtsp_url

    # Verificar se ffmpeg está disponível
    global _ffmpeg_available
    _ffmpeg_available = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True).returncode == 0
    if _ffmpeg_available:
        print("FFmpeg disponível  : stream via HTTP-FLV (baixa latência)")
    else:
        print("[AVISO] ffmpeg não encontrado — stream via MJPEG (fallback)")
        print("        Para HTTP-FLV: adicione BR2_PACKAGE_FFMPEG=y ao defconfig")

    # Web server (daemon thread)
    web_server = ThreadedHTTPServer(("0.0.0.0", args.web_port), Handler)
    web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
    web_thread.start()
    print(f"  Web UI      : http://0.0.0.0:{args.web_port}/")

    global _persist_detections
    _persist_detections = args.persist_detections

    print("\nTransmitindo... (Ctrl+C para parar)\n")

    # Main camera + RTSP loop (blocks until signal).
    # A thread de inferência é gerida dentro de camera_loop.
    camera_loop(args, rtsp)

    # Teardown
    _status = "Stopped"
    web_server.shutdown()
    with _detector_lock:
        if _detector is not None:
            try:
                _detector.close()
            except Exception:
                pass
    del rtsp

    print(f"\nFinalizado. {_frame_count} frames  "
          f"cam={_cam_fps:.1f}fps  inf={_infer_fps:.1f}fps  "
          f"dets={_det_total}")


if __name__ == "__main__":
    main()
