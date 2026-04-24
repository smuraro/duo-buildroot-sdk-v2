#!/usr/bin/env python3
"""
sample_rtsp_webserver.py — Câmera + inferência TDL + RTSP + interface web

Transmite o vídeo com overlay de detecções via RTSP (H264/H265) e serve
uma interface web que exibe o stream em tempo real via HTTP-FLV (mpegts.js).
O FFmpeg atua como proxy interno RTSP→FLV sem re-encoding.

Requer: ffmpeg disponível no PATH do dispositivo.

Suporta três fontes de vídeo (--input):
  vazio         câmera VI local (padrão)
  rtsp://...    stream RTSP decodificado por hardware (VDEC)
  usb / usb:N   câmera USB /dev/video0 (ou /dev/videoN)

Uso (câmera VI — model type auto-detectado):
    python3 sample_rtsp_webserver.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel

Uso (câmera VI — parâmetros explícitos):
    python3 sample_rtsp_webserver.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        --model-type SCRFD_DET_FACE [--width 1280] [--height 720]

Uso (entrada RTSP — resolução auto-detectada):
    python3 sample_rtsp_webserver.py \\
        --model ... --input rtsp://192.168.1.10:554/live [--transport tcp|udp]

Uso (câmera USB):
    python3 sample_rtsp_webserver.py \\
        --model ... --input usb

Pipeline dois estágios (ex: SCRFD → KEYPOINT_FACE_V2):
    python3 sample_rtsp_webserver.py \\
        --model /root/cv181x/keypoint_face_v2_64_64_INT8_cv181x.cvimodel \\
        --model-type KEYPOINT_FACE_V2 \\
        --stage1-model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        --stage1-model-type SCRFD_DET_FACE
Na interface web, ao selecionar um detector facial, a opção de stage-2
(keypoints/landmarks/atributos faciais) aparece automaticamente.

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


# ─── Auto-detecção de modelo e resolução ─────────────────────────────────────

def _detect_model_type(model_path):
    """Infere o ModelType a partir do nome do arquivo .cvimodel."""
    basename = os.path.basename(model_path).lower().replace("_", "")
    _auto_map = [
        ("keypointyolov8poseperson17", "KEYPOINT_YOLOV8POSE_PERSON17"),
        ("scrfddetface",        "SCRFD_DET_FACE"),
        ("yolov8detcoco80",     "YOLOV8_DET_COCO80"),
        ("yolov8ndetcoco80",    "YOLOV8_DET_COCO80"),
        ("yolov11ndetcoco80",   "YOLOV11N_DET_COCO80"),
        ("yolo11ndetcoco80",    "YOLOV11N_DET_COCO80"),
        ("yolo11detcoco80",     "YOLOV11N_DET_COCO80"),
        ("yolo26detcoco80",     "YOLOV26_DET_COCO80"),
        ("yoloxdetcoco80",      "YOLOX_DET_COCO80"),
        ("yolov10detcoco80",    "YOLOV10_DET_COCO80"),
        ("yolov7detcoco80",     "YOLOV7_DET_COCO80"),
        ("yolov6detcoco80",     "YOLOV6_DET_COCO80"),
        ("yolov5detcoco80",     "YOLOV5_DET_COCO80"),
    ]
    for pattern, enum_name in _auto_map:
        if pattern in basename:
            return enum_name
    return None


def _probe_rtsp_resolution(url, transport="tcp"):
    """Conecta brevemente ao stream RTSP via OpenCV para descobrir a resolução.

    Retorna (width, height) ou (0, 0) se não conseguir.
    """
    try:
        import cv2
    except ImportError:
        return 0, 0
    try:
        env_key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        old = os.environ.get(env_key, "")
        os.environ[env_key] = f"rtsp_transport;{transport}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if old:
            os.environ[env_key] = old
        else:
            os.environ.pop(env_key, None)

        if not cap.isOpened():
            return 0, 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 0, 0


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
            # Two-stage face + CLS_ATTRIBUTE: has bbox AND attribute scores
            if "x1" in first and any(
                    k.endswith("_score") for k in first):
                # The thumbnail painted in the previous frame persists in the
                # VB buffer that the detector saw this frame, so SCRFD can
                # latch onto it as a second face. Drop any detection whose
                # center falls inside the thumbnail rectangle.
                tx1, ty1, tx2, ty2 = image.get_thumbnail_rect(
                    frame, thumb_size=96)
                dets = [d for d in dets
                        if not (tx1 <= 0.5 * (d["x1"] + d["x2"]) < tx2 and
                                ty1 <= 0.5 * (d["y1"] + d["y2"]) < ty2)]
                if not dets:
                    return
                first = dets[0]
                # Capture the pristine face pixels BEFORE any bbox/label draw
                # contaminates the frame, then paint the preview LAST so the
                # thumbnail shows exactly what the stage-2 model saw.
                snapshot = image.capture_face_crop(
                    frame, first["x1"], first["y1"],
                    first["x2"], first["y2"], thumb_size=96)
                image.draw_detections(frame, dets, score_threshold=threshold)
                image.draw_classification(frame, dets)
                image.draw_face_thumbnail(frame, snapshot)
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

_infer_fps       = 0.0
_infer_ms        = 0.0
_infer_ms_acc    = 0.0      # acumulador para média por janela de reporte
_infer_count_window = 0     # contagem de inferências na janela actual
_cam_fps         = 0.0
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

# Stage-1 detector (face detector for two-stage pipelines, e.g. KEYPOINT_FACE_V2).
# Protected by _detector_lock — never accessed outside that lock in the
# inference worker so no separate lock is needed.
_stage1_detector    = None
_stage1_model_type  = ""
_stage1_switch_status = ""

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
_source_type   = "vi"   # "vi" | "vdec" | "usb"


def _inference_worker(cam, vb_sem):
    global _running, _infer_fps, _infer_ms, _last_detect_frame, _source_type
    global _infer_ms_acc, _infer_count_window
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
            # Mantém _detector_lock durante toda a inferência (stage1 + stage2).
            # _do_switch_model/_do_switch_stage1 chamam old.close() FORA do lock,
            # então só executam depois que este bloco terminar — garantindo que o
            # NPU runtime nunca é liberado enquanto inference() o está a usar.
            with _detector_lock:
                det    = _detector
                stage1 = _stage1_detector
                if det is not None:
                    try:
                        mt = _current_model_type.upper()
                        use_two_stage = (stage1 is not None and
                                         ("KEYPOINT" in mt or "LANDMARK" in mt
                                          or "CLS_ATTRIBUTE" in mt))
                        if use_two_stage:
                            # Pipeline dois estágios: faces → landmarks
                            faces = stage1.inference(frame)
                            dets  = det.inference_with_detections(frame, faces) \
                                    if faces else []
                        else:
                            dets = det.inference(frame)
                        # Use C++ steady_clock measurement (excludes GIL wait time).
                        _infer_ms = det.get_last_inference_ms()
                        _infer_ms_acc += _infer_ms
                        _infer_count_window += 1
                        with _det_lock:
                            if isinstance(dets, (list, tuple)):
                                _last_detections[:] = dets
                            else:
                                _last_detections[:] = [dets] if dets else []
                            if dets:
                                _last_detect_frame = _frame_count
                    except Exception as e:
                        print(f"[AVISO] inference error: {e}")

                    count += 1
                    elapsed = time.time() - t0
                    if elapsed > 0:
                        _infer_fps = count / elapsed

        # Libera de acordo com a fonte:
        # vi:   cam.release() FIFO + semáforo
        # vdec: release_inference() (slot pinado pelo camera_loop)
        # usb:  cam.release() no-op; sem semáforo
        if _source_type == "vi":
            cam.release()
            vb_sem.release()
        elif _source_type == "vdec":
            cam.release_inference()
        else:
            cam.release()


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


def _do_switch_stage1(model_type_name: str, model_path: str) -> tuple:
    """Load (or clear) the stage-1 face detector used in two-stage pipelines.

    Pass empty strings for both arguments to disable the stage-1 detector.
    Returns (ok: bool, message: str).
    """
    global _stage1_detector, _stage1_model_type, _stage1_switch_status

    if not model_type_name and not model_path:
        # Clear stage-1
        with _detector_lock:
            old = _stage1_detector
            _stage1_detector   = None
            _stage1_model_type = ""
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        _stage1_switch_status = "Stage-1 desativado"
        print("[switch_stage1] desativado")
        return True, _stage1_switch_status

    model_type = getattr(nn.ModelType, model_type_name, None)
    if model_type is None:
        return False, f"Stage-1 ModelType desconhecido: {model_type_name}"
    if not os.path.isfile(model_path):
        return False, f"Stage-1 arquivo não encontrado: {model_path}"

    try:
        new_det = nn.get_model(model_type, model_path)
        new_det.set_threshold(_current_threshold)
    except Exception as e:
        return False, f"Erro ao carregar stage-1: {e}"

    with _detector_lock:
        old                = _stage1_detector
        _stage1_detector   = new_det
        _stage1_model_type = model_type_name

    if old is not None:
        try:
            old.close()
        except Exception:
            pass

    with _det_lock:
        _last_detections.clear()

    msg = f"Stage-1 trocado: {model_type_name} / {os.path.basename(model_path)}"
    _stage1_switch_status = msg
    print(f"[switch_stage1] {msg}")
    return True, msg


# ─── Main camera + RTSP loop ──────────────────────────────────────────────────


def _open_source(args):
    """Abre a fonte de vídeo conforme --input e retorna (cam, source_type)."""
    inp = args.input.strip()
    if inp.lower().startswith("usb"):
        device = 0
        if ":" in inp:
            try:
                device = int(inp.split(":", 1)[1])
            except ValueError:
                pass
        if not hasattr(image, "UsbCamera"):
            print("[ERRO] UsbCamera não disponível nesta build (requer OpenCV videoio).")
            sys.exit(1)
        req_w = args.width if args.width else 640
        req_h = args.height if args.height else 480
        print(f"\nAbrindo câmera USB /dev/video{device}  {req_w}x{req_h} ...")
        cam = image.UsbCamera(device, req_w, req_h)
        if not cam.is_opened():
            print(f"[ERRO] Não foi possível abrir /dev/video{device}.")
            sys.exit(1)
        if hasattr(cam, 'width') and hasattr(cam, 'height'):
            args.width, args.height = cam.width, cam.height
        else:
            args.width, args.height = req_w, req_h
        print(f"  Backend: USB V4L2 /dev/video{device}  {args.width}x{args.height}")
        return cam, "usb"
    elif inp.startswith("rtsp://") or inp.startswith("rtsps://"):
        w, h = args.width, args.height
        if w == 0 or h == 0:
            print(f"\nAuto-detectando resolução de {inp} ...")
            w, h = _probe_rtsp_resolution(inp, args.transport)
            if w > 0 and h > 0:
                print(f"  Resolução detectada: {w}x{h}")
                args.width, args.height = w, h
            else:
                print("[ERRO] Não foi possível detectar resolução do stream RTSP.\n"
                      "  Especifique manualmente com --width e --height.")
                sys.exit(1)
        print(f"\nAbrindo stream RTSP {inp} ({args.transport}) ...")
        cam = image.RtspClientVdec(inp, width=w, height=h,
                                   transport=args.transport)
        if not cam.is_opened():
            print("[ERRO] Não foi possível abrir o stream RTSP.")
            sys.exit(1)
        print(f"  Backend: VDEC hardware (H264)  {w}x{h}")
        return cam, "vdec"
    else:
        if args.width == 0:
            args.width = 1280
        if args.height == 0:
            args.height = 720
        print(f"\nAbrindo câmera VI {args.width}x{args.height} ...")
        cam = image.Camera(args.width, args.height, image.ImageFormat.YUV420SP_VU,
                           vb_buffer_num=_VB_BUFFER_NUM,
                           mirror=args.mirror, flip=args.flip)
        print("  Backend: câmera VI local")
        return cam, "vi"


def camera_loop(args, rtsp):
    global _running, _cam_fps, _frame_count, _det_total, _status, _source_type
    global _infer_ms_acc, _infer_count_window

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

    cam, _source_type = _open_source(args)

    # Semáforo apenas para câmera VI (VB pool limitado).
    vb_sem = threading.Semaphore(_VB_BUFFER_NUM - 2)

    # Thread de inferência/release iniciada aqui, depois de cam ser criada,
    # para que possa chamar cam.release() em ordem FIFO.
    infer_thread = threading.Thread(
        target=_inference_worker, args=(cam, vb_sem), daemon=True)
    infer_thread.start()

    try:
        while _running and (limit is None or frame_idx < limit):
            # Semáforo apenas para VI; VDEC e USB gerenciam própria memória.
            if _source_type == "vi" and not vb_sem.acquire(timeout=0.5):
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

            # Para RtspClientVdec: pina o frame para a thread de inferência e
            # libera o slot de display antes do próximo read().
            if _source_type == "vdec":
                cam.pin_for_inference()

            # Enfileira APÓS rtsp.send_frame e frame_to_jpeg.
            do_infer = (frame_idx % skip_every == 0)
            with _release_lock:
                if len(_release_queue) >= _MAX_RELEASE_PENDING:
                    do_infer = False
                # USB: cam.release() é no-op, então frames sem inferência
                # não precisam da fila — VB blocks liberados pelo GC quando
                # a referência local 'frame' é sobrescrita no próximo read().
                if do_infer or _source_type != "usb":
                    _release_queue.append((frame, do_infer))

            if _source_type == "vdec":
                cam.release()  # libera slot de display; inferência usa slot pinado

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
                ni = max(_infer_count_window, 1)
                avg_infer = _infer_ms_acc / ni
                print(f"  frame {frame_idx:6d}  "
                      f"cam={_cam_fps:5.1f}fps  "
                      f"inf={_infer_fps:4.1f}fps  "
                      f"dets={len(dets)}  "
                      f"| read={_cam_ms_acc/n:5.1f}ms  "
                      f"inf={avg_infer:5.1f}ms  "
                      f"draw={_draw_ms_acc/n:4.1f}ms  "
                      f"send={_send_ms_acc/n:5.1f}ms")
                _infer_ms_acc = 0.0
                _infer_count_window = 0
                _cam_ms_acc = _draw_ms_acc = _send_ms_acc = 0.0
                _acc_count  = 0
                t_report    = now

    except Exception as exc:
        _status = f"Erro: {exc}"
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        if _source_type == "vi":
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
        <div id="stage2-wrap" style="display:none;flex-direction:column;gap:8px;margin-top:4px;padding-top:8px;border-top:1px solid var(--border)">
          <div style="font-family:var(--mono);font-size:10px;color:var(--accent2);letter-spacing:.05em">STAGE-2 (Landmarks / Keypoints / Atributos)</div>
          <select id="sel-s2-type" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:5px 6px;font-family:var(--mono);font-size:11px;border-radius:3px">
            <option value="">-- sem stage-2 --</option>
          </select>
          <select id="sel-s2-file" style="background:var(--bg);color:var(--text);border:1px solid var(--border);padding:5px 6px;font-family:var(--mono);font-size:11px;border-radius:3px">
            <option value="">-- cvimodel --</option>
          </select>
        </div>
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

// ─── Model selector (unified: main + optional stage-2) ──────────────────────

// Face detectors that can serve as stage-1 for KEYPOINT/LANDMARK pipelines
function isFaceDetector(t) {
  return /SCRFD|RETINAFACE/.test(t) && /FACE/.test(t);
}
function isStage2Model(t) {
  return /KEYPOINT|LANDMARK|CLS_ATTRIBUTE/.test(t);
}

let _allModelTypes = [];

async function loadModelSelector() {
  try {
    _allModelTypes = await fetch('/api/models').then(r => r.json());
    const selType = document.getElementById('sel-type');
    selType.innerHTML = '<option value="">-- model type --</option>';
    _allModelTypes.forEach(t => {
      const o = document.createElement('option');
      o.value = o.textContent = t;
      selType.appendChild(o);
    });
    selType.onchange = () => onTypeChange(selType.value);
  } catch(e) {
    document.getElementById('switch-msg').textContent = 'Erro ao carregar tipos';
  }
}

async function _populateFileSelector(selFile, modelType) {
  if (!modelType) {
    selFile.innerHTML = '<option value="">-- cvimodel --</option>';
    return;
  }
  selFile.innerHTML = '<option value="">Buscando...</option>';
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

function onTypeChange(modelType) {
  _populateFileSelector(document.getElementById('sel-file'), modelType);
  document.getElementById('switch-msg').textContent = '';

  // Show/hide stage-2 section
  const wrap = document.getElementById('stage2-wrap');
  if (isFaceDetector(modelType)) {
    wrap.style.display = 'flex';
    // Populate stage-2 type selector with KEYPOINT/LANDMARK models
    const selS2 = document.getElementById('sel-s2-type');
    selS2.innerHTML = '<option value="">-- sem stage-2 --</option>';
    _allModelTypes.filter(isStage2Model).forEach(t => {
      const o = document.createElement('option');
      o.value = o.textContent = t;
      selS2.appendChild(o);
    });
    selS2.onchange = () => _populateFileSelector(
      document.getElementById('sel-s2-file'), selS2.value);
    // Reset stage-2 file
    document.getElementById('sel-s2-file').innerHTML = '<option value="">-- cvimodel --</option>';
  } else {
    wrap.style.display = 'none';
    document.getElementById('sel-s2-type').value = '';
    document.getElementById('sel-s2-file').innerHTML = '<option value="">-- cvimodel --</option>';
  }
}

async function doSwitch() {
  const mainType = document.getElementById('sel-type').value;
  const mainPath = document.getElementById('sel-file').value;
  const msg      = document.getElementById('switch-msg');
  const btn      = document.getElementById('btn-switch');

  if (!mainType || !mainPath) {
    msg.style.color = 'var(--warn)';
    msg.textContent = 'Selecione tipo e arquivo';
    return;
  }

  const s2Type = document.getElementById('sel-s2-type').value;
  const s2Path = document.getElementById('sel-s2-file').value;
  const hasTwoStage = isFaceDetector(mainType) && s2Type && s2Path;

  if (isFaceDetector(mainType) && s2Type && !s2Path) {
    msg.style.color = 'var(--warn)';
    msg.textContent = 'Selecione o arquivo do stage-2';
    return;
  }

  btn.disabled    = true;
  btn.textContent = 'Aguarde...';
  msg.style.color = 'var(--dim)';
  msg.textContent = hasTwoStage ? 'Carregando pipeline...' : 'Carregando...';

  try {
    const payload = hasTwoStage
      ? { model_type: s2Type, model_path: s2Path,
          stage1_model_type: mainType, stage1_model_path: mainPath }
      : { model_type: mainType, model_path: mainPath };

    const r = await fetch('/api/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
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
                        "emotion":  lambda d, s: f"Emotion {d.get('emotion', f'{s:.2f}')}",
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
                "stage1_model_type": _stage1_model_type,
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
                req = json.loads(body)
                ok, msg = _do_switch_model(req.get("model_type", ""),
                                           req.get("model_path", ""))
                if ok:
                    s1_type = req.get("stage1_model_type", "")
                    s1_path = req.get("stage1_model_path", "")
                    if s1_type and s1_path:
                        ok2, msg2 = _do_switch_stage1(s1_type, s1_path)
                        if ok2:
                            msg += " + " + msg2
                        else:
                            msg += " (stage-1 falhou: " + msg2 + ")"
                    else:
                        # Limpa stage-1 quando não há pipeline dois estágios
                        _do_switch_stage1("", "")
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
                   help="Nome do ModelType (ex: SCRFD_DET_FACE). "
                        "Auto-detectado pelo nome do arquivo se omitido; "
                        "pode ser selecionado na interface web.")
    p.add_argument("--width",     type=int, default=0,
                   help="Largura em pixels (auto-detectado se omitido; "
                        "padrão: 1280 para VI, 640 para USB)")
    p.add_argument("--height",    type=int, default=0,
                   help="Altura em pixels (auto-detectado se omitido; "
                        "padrão: 720 para VI, 480 para USB)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--codec",     default="h264", choices=["h264", "h265"])
    p.add_argument("--bitrate",   type=int, default=3072,
                   help="Bitrate de codificação em kbps (padrão: 3072). "
                        "Aumente para melhor qualidade em movimento (ex: 4096, 6144).")
    p.add_argument("--fps",       type=int, default=15,
                   help="Frame rate declarado ao encoder VENC (padrão: 15). "
                        "Deve refletir o FPS real da câmera — câmeras USB lentas podem exigir "
                        "valores como 5 ou 10; valor errado causa GOP incorreto e stream estático.")
    p.add_argument("--gop",       type=int, default=0,
                   help="Intervalo de keyframe em frames (padrão: 0 = automático: 1× fps). "
                        "Use 1 para câmeras USB lentas (todo frame é I-frame).")
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
    p.add_argument("--input",     default="",
                   help="Fonte de vídeo: vazio = câmera VI local; "
                        "rtsp://... = stream RTSP (VDEC hardware); "
                        "usb = /dev/video0; usb:1 = /dev/video1.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "udp"],
                   help="Transporte RTSP de entrada (padrão: tcp)")
    p.add_argument("--mirror",    action="store_true", default=False,
                   help="Espelhar horizontalmente (flip esquerda↔direita). Apenas câmera VI.")
    p.add_argument("--flip",      action="store_true", default=False,
                   help="Inverter verticalmente (flip cima↔baixo). Apenas câmera VI.")
    p.add_argument("--labels",    default="", dest="labels",
                   help="Nomes das classes: arquivo .txt ou 'cls0,cls1,...'")
    p.add_argument("--stage1-model", default="", dest="stage1_model",
                   help="Modelo do estágio 1 (ex: SCRFD) para pipeline dois estágios. "
                        "Necessário para KEYPOINT/LANDMARK/CLS_ATTRIBUTE; pode ser "
                        "configurado na interface web.")
    p.add_argument("--stage1-model-type", default="SCRFD_DET_FACE",
                   dest="stage1_model_type",
                   help="ModelType do estágio 1 (padrão: SCRFD_DET_FACE).")
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

    # --- Model type (auto-detect se --model fornecido sem --model-type) ---
    model_type_name = args.model_type
    if args.model and not model_type_name:
        model_type_name = _detect_model_type(args.model)
        if model_type_name:
            print(f"ModelType auto     : {model_type_name}")
        else:
            print("[AVISO] Não foi possível inferir --model-type pelo nome do arquivo.\n"
                  "        Iniciando sem modelo — selecione na interface web.")

    # Modelo inicial (opcional)
    if args.model and model_type_name:
        if args.labels:
            _custom_labels = _load_labels(args.labels)
            print(f"Labels carregados  : {len(_custom_labels)} classes (--labels)")
        else:
            _custom_labels = _labels_from_factory(model_type_name)
            if _custom_labels:
                print(f"Labels carregados  : {len(_custom_labels)} classes (model_factory.json)")

        model_type = getattr(nn.ModelType, model_type_name, None)
        if model_type is None:
            print(f"[ERRO] ModelType desconhecido: {model_type_name}")
            print("  Tipos disponíveis: " + ", ".join(
                t for t in dir(nn.ModelType) if not t.startswith("_")))
            sys.exit(1)

        print(f"Carregando modelo  : {args.model}")
        print(f"ModelType          : {model_type_name}")
        detector = nn.get_model(model_type, args.model)
        detector.set_threshold(args.threshold)
        print(f"Limiar             : {detector.get_threshold():.2f}")

        with _detector_lock:
            _detector           = detector
            _current_model_type = model_type_name
    elif not args.model:
        print("Nenhum modelo inicial — selecione na interface web.")

    # Stage-1 detector inicial (opcional — pipeline dois estágios)
    if args.stage1_model:
        ok, msg = _do_switch_stage1(args.stage1_model_type, args.stage1_model)
        if not ok:
            print(f"[ERRO] {msg}")
            sys.exit(1)

    # --- Fonte de vídeo (aberta antes do RTSP para obter resolução auto-detectada) ---
    global _source_type
    _source_type_init = args.input.strip().lower()

    # Auto-ajuste de fps/gop para câmera USB
    fps = args.fps
    bitrate = args.bitrate
    if _source_type_init.startswith("usb") and fps > 10:
        fps = 3
        print(f"  [auto] FPS ajustado para {fps} (câmera USB geralmente entrega ≤5fps)")
    if _source_type_init.startswith("usb") and args.bitrate >= 2048:
        bitrate = 1024
        print(f"  [auto] Bitrate ajustado para {bitrate}kbps (USB: NALUs menores → "
              f"compatível com VLC/UDP)")
    gop = args.gop if args.gop > 0 else max(1, fps)
    if _source_type_init.startswith("usb") and args.gop <= 0:
        gop = max(1, fps)
        print(f"  [auto] GOP ajustado para {gop} (1 keyframe/s para câmera USB)")

    # RTSP server (args.width/height pode ter sido atualizado pelo _open_source em camera_loop,
    # mas para isso funcionar com auto-detect, precisamos abrir a fonte primeiro em camera_loop
    # e usar as dimensões detectadas. Como a resolução precisa ser conhecida antes do RTSP server,
    # fazemos o probe/default aqui.)
    # Aplica defaults se width/height ainda são 0
    if args.width == 0 or args.height == 0:
        inp = args.input.strip()
        if inp.lower().startswith("usb"):
            if args.width == 0: args.width = 640
            if args.height == 0: args.height = 480
        elif inp.startswith("rtsp://") or inp.startswith("rtsps://"):
            print(f"\nAuto-detectando resolução de {inp} ...")
            pw, ph = _probe_rtsp_resolution(inp, args.transport)
            if pw > 0 and ph > 0:
                args.width, args.height = pw, ph
                print(f"  Resolução detectada: {args.width}x{args.height}")
            else:
                print("[ERRO] Não foi possível detectar resolução do stream RTSP.\n"
                      "  Especifique manualmente com --width e --height.")
                sys.exit(1)
        else:
            if args.width == 0: args.width = 1280
            if args.height == 0: args.height = 720

    print(f"\nIniciando servidor RTSP {args.width}x{args.height} "
          f"codec={args.codec} bitrate={bitrate}kbps fps={fps} "
          f"gop={gop} ({gop/fps:.1f}s) sessão={args.session} ...")
    rtsp = image.RTSPServer(args.width, args.height, chn=0,
                            codec=args.codec, session_name=args.session,
                            bitrate=bitrate, gop=gop, fps=fps)
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
        if _stage1_detector is not None:
            try:
                _stage1_detector.close()
            except Exception:
                pass
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
