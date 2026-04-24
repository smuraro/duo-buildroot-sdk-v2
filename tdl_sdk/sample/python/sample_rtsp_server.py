#!/usr/bin/env python3
"""
sample_rtsp_server.py — Câmera + detecção de objetos + servidor RTSP

Captura frames continuamente numa thread dedicada e transmite ao cliente
RTSP à taxa máxima da câmera. A inferência corre em paralelo; o overlay de
detecções é actualizado a cada novo resultado sem bloquear o stream.

Suporta três fontes de vídeo (--input):
  vazio         câmera VI local (padrão)
  rtsp://...    stream RTSP decodificado por hardware (VDEC)
  usb / usb:N   câmera USB /dev/video0 (ou /dev/videoN)

Uso (câmera VI — model type auto-detectado):
    python3 sample_rtsp_server.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel

Uso (câmera VI — model type explícito):
    python3 sample_rtsp_server.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        --model-type SCRFD_DET_FACE [--width 1280] [--height 720]

Uso (entrada RTSP — resolução auto-detectada):
    python3 sample_rtsp_server.py \\
        --model ... --input rtsp://192.168.1.10:554/live [--transport tcp|udp]

Uso (câmera USB):
    python3 sample_rtsp_server.py \\
        --model ... --input usb

Pipeline dois estágios (ex: SCRFD → KEYPOINT_FACE_V2):
    python3 sample_rtsp_server.py \\
        --model /root/cv181x/keypoint_face_v2_64_64_INT8_cv181x.cvimodel \\
        --model-type KEYPOINT_FACE_V2 \\
        --stage1-model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        --stage1-model-type SCRFD_DET_FACE

Conectar ao stream:
    vlc rtsp://<ip-do-dispositivo>:554/<session>
    ffplay rtsp://<ip-do-dispositivo>:554/<session>
"""

import argparse
import signal
import sys
import threading
import time

import tdl
from tdl import image, nn


# ─── Auto-detecção de modelo e resolução ─────────────────────────────────────

def _detect_model_type(model_path):
    """Infere o ModelType a partir do nome do arquivo .cvimodel."""
    import os
    basename = os.path.basename(model_path).lower().replace("_", "")
    _auto_map = [
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
        import os
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


# ─── Labels customizados ──────────────────────────────────────────────────────

_custom_labels: dict = {}


def _labels_from_factory(model_type_name: str) -> dict:
    """Return {class_id: name} from model_factory.json via nn.get_model_types().

    Returns empty dict if no types are defined for the model type.
    """
    try:
        types = nn.get_model_types(model_type_name)
        return {i: n for i, n in enumerate(types)} if types else {}
    except Exception:
        return {}


def _load_labels(labels_arg: str) -> dict:
    """Parse --labels into a dict {class_id: name}.

    Accepts:
      - path to a text file: one label per line, index = line number
      - comma-separated string: "cat,dog,bird"
    Returns empty dict if labels_arg is None/empty.
    """
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
    """Custom labels → original name."""
    if (class_name.startswith("cls") or class_name == "UNDEFINED") \
            and class_id in _custom_labels:
        return _custom_labels[class_id]
    return class_name


def _draw_inference(frame, dets, is_keypoint: bool, threshold: float):
    """Dispatch inference result to the appropriate draw function."""
    import numpy as np
    if dets is None or isinstance(dets, np.ndarray) or not dets:
        return
    if isinstance(dets, (list, tuple)):
        first = dets[0]
        if isinstance(first, str):
            image.draw_ocr(frame, dets)
            return
        if isinstance(first, dict):
            if "output_width" in first:
                image.draw_segmentation(frame, dets)
                return
            if "bboxes_seg" in first or "mask_width" in first:
                image.draw_instance_segmentation(frame, dets,
                                                 score_threshold=threshold)
                return
            if "landmarks" in first and "x1" not in first:
                image.draw_keypoints(frame, dets, score_threshold=threshold)
                return
            # Two-stage face + CLS_ATTRIBUTE: has bbox AND attribute scores
            if "x1" in first and any(
                    k.endswith("_score") for k in first):
                # Previous frame's thumbnail persists in the VB buffer, so the
                # detector can re-detect it as a second face. Drop detections
                # whose center lies inside the thumbnail rectangle.
                tx1, ty1, tx2, ty2 = image.get_thumbnail_rect(
                    frame, thumb_size=96)
                dets = [d for d in dets
                        if not (tx1 <= 0.5 * (d["x1"] + d["x2"]) < tx2 and
                                ty1 <= 0.5 * (d["y1"] + d["y2"]) < ty2)]
                if not dets:
                    return
                first = dets[0]
                # Capture pristine face pixels BEFORE bbox/label drawing, then
                # paint the preview LAST to show exactly what stage-2 saw.
                snapshot = image.capture_face_crop(
                    frame, first["x1"], first["y1"],
                    first["x2"], first["y2"], thumb_size=96)
                image.draw_detections(frame, dets, score_threshold=threshold)
                image.draw_classification(frame, dets)
                image.draw_face_thumbnail(frame, snapshot)
                return
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
            enriched = [{**d, "class_name": _resolve_name(
                            d.get("class_id", -1), d.get("class_name", ""))}
                        for d in dets]
            if "landmarks" in first:
                image.draw_detections(frame, enriched, score_threshold=threshold)
                image.draw_keypoints(frame, enriched, score_threshold=threshold)
            elif is_keypoint:
                image.draw_keypoints(frame, enriched, score_threshold=threshold)
            else:
                image.draw_detections(frame, enriched, score_threshold=threshold)
    elif isinstance(dets, dict):
        if "output_width" in dets:
            image.draw_segmentation(frame, dets)
        elif "bboxes_seg" in dets:
            image.draw_instance_segmentation(frame, dets, score_threshold=threshold)
        elif "class_id" in dets or "is_male" in dets:
            image.draw_classification(frame, dets)


# ─── Sinalização de encerramento ──────────────────────────────────────────────

_running = True


def _sigint_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# ─── Argumentos ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Câmera + detecção de objetos + servidor RTSP")
    p.add_argument("--model", required=True,
                   help="Caminho para o arquivo .cvimodel")
    p.add_argument("--model-type", default="",
                   help="Nome do ModelType (ex: SCRFD_DET_FACE, YOLOV8_DET_COCO80). "
                        "Auto-detectado pelo nome do arquivo se omitido.")
    p.add_argument("--width",  type=int, default=0,
                   help="Largura em pixels (auto-detectado se omitido; "
                        "padrão: 1280 para VI, 640 para USB)")
    p.add_argument("--height", type=int, default=0,
                   help="Altura em pixels (auto-detectado se omitido; "
                        "padrão: 720 para VI, 480 para USB)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Limiar de confiança (padrão: 0.5)")
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
    p.add_argument("--codec", default="h264", choices=["h264", "h265"],
                   help="Codec de vídeo: h264 ou h265 (padrão: h264)")
    p.add_argument("--session", default="live",
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
    p.add_argument("--input", default="",
                   help="Fonte de vídeo: vazio = câmera VI local; "
                        "rtsp://... = stream RTSP (VDEC hardware); "
                        "usb = /dev/video0; usb:1 = /dev/video1.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "udp"],
                   help="Transporte RTSP de entrada (padrão: tcp)")
    p.add_argument("--mirror", action="store_true", default=False,
                   help="Espelhar horizontalmente a imagem da câmera (flip esquerda↔direita). Apenas câmera VI.")
    p.add_argument("--flip",  action="store_true", default=False,
                   help="Inverter verticalmente a imagem da câmera (flip cima↔baixo). Apenas câmera VI.")
    p.add_argument("--labels", default="", dest="labels",
                   help="Nomes das classes para modelos genéricos (YOLOV26, YOLOV8…). "
                        "Aceita caminho para arquivo .txt (uma classe por linha) "
                        "ou lista separada por vírgula: 'gato,cachorro,pássaro'.")
    p.add_argument("--stage1-model", default="", dest="stage1_model",
                   help="Modelo do estágio 1 (ex: SCRFD) para pipeline dois estágios. "
                        "Necessário para modelos como KEYPOINT_FACE_V2 que requerem "
                        "detecções faciais como entrada.")
    p.add_argument("--stage1-model-type", default="SCRFD_DET_FACE",
                   dest="stage1_model_type",
                   help="ModelType do estágio 1 (padrão: SCRFD_DET_FACE).")
    return p.parse_args()


# ─── Thread de inferência / release ──────────────────────────────────────────
#
# Todos os frames lidos passam pela _release_queue antes de serem devolvidos
# à câmera.  A thread processa-os em ordem FIFO: corre inferência nos marcados
# com do_infer=True, depois chama sempre cam.release().
#
# Por que isso é necessário:
#   ViDecoder::release() usa uma fila interna (frameQueues) e devolve sempre o
#   frame mais antigo — se o main thread chamasse cam.release() para o frame N+1
#   enquanto a thread de inferência ainda usa o frame N, a liberação cairia sobre
#   o VB block errado → "vb released" no VPSS do preprocessador.
#
# Regra: cam.release() só é chamado dentro desta thread, garantindo ordem FIFO.

_release_queue      = []       # lista de (frame, do_infer: bool)
_release_lock       = threading.Lock()
_last_detections    = []
_det_lock           = threading.Lock()
_infer_fps          = 0.0
_infer_ms           = 0.0     # último tempo de inferência em ms
_infer_ms_acc       = 0.0     # acumulador para média por janela de reporte
_infer_count_window = 0       # contagem de inferências na janela atual
_persist_detections = False   # configurado em main() via args.persist_detections
_last_detect_frame  = -1      # frame_idx da última inferência positiva
_frame_count        = 0       # frame atual (atualizado pelo loop principal)

# Máximo de frames aguardando na fila de release.  Se a fila encher (inferência
# mais lenta que a câmera), novos frames entram com do_infer=False para não
# acumular mais de MAX_RELEASE_PENDING VB blocks além do que está em inferência.
_MAX_RELEASE_PENDING = 2

# Pool de VB blocks da câmera.  Com inferência assíncrona, o main thread e
# a thread de inferência seguram frames simultaneamente.  O semáforo impede
# que cam.read() avance quando todos os blocks estão ocupados, evitando o
# stall do ISP (erro "jobs wait(0) work(0) done(0)").
#
#   _VB_BUFFER_NUM  = tamanho do pool criado no driver (passado ao Camera())
#   _vb_sem inicial = _VB_BUFFER_NUM - 2
#     (reserva 1 block para o VPSS output interno + 1 de folga)
_VB_BUFFER_NUM = 5
_vb_sem        = threading.Semaphore(_VB_BUFFER_NUM - 2)
_source_type   = "vi"   # "vi" | "vdec" | "usb"


def _inference_worker(detector, cam, threshold, stage1_detector=None):
    """Thread de inferência/release.

    Se stage1_detector for fornecido, executa pipeline dois estágios:
      1. stage1_detector.inference(frame)  → detecções (ex: faces do SCRFD)
      2. detector.inference_with_detections(frame, faces) → landmarks por face
    Caso contrário executa inferência simples: detector.inference(frame).
    """
    global _running, _infer_fps, _infer_ms, _infer_ms_acc, _infer_count_window
    global _last_detect_frame, _source_type
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
            ti = time.time()
            if stage1_detector is not None:
                faces = stage1_detector.inference(frame)
                dets  = detector.inference_with_detections(frame, faces) if faces else []
            else:
                dets = detector.inference(frame)
            # Use C++ steady_clock measurement (excludes GIL wait time).
            dt = detector.get_last_inference_ms()
            _infer_ms = dt
            _infer_ms_acc += dt
            _infer_count_window += 1

            with _det_lock:
                if isinstance(dets, (list, tuple)):
                    _last_detections[:] = dets
                else:
                    _last_detections[:] = [dets] if dets else []
                if dets:
                    _last_detect_frame = _frame_count

            count += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                _infer_fps = count / elapsed

        # Libera de acordo com a fonte:
        # vi:   cam.release() + semáforo (FIFO, VB pool)
        # vdec: release_inference() (slot de inferência pinado pelo loop principal)
        # usb:  cam.release() no-op; VPSSImage usa ION, sem semáforo
        if _source_type == "vi":
            cam.release()
            _vb_sem.release()
        elif _source_type == "vdec":
            cam.release_inference()
        else:
            cam.release()


# ─── Main ─────────────────────────────────────────────────────────────────────

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
        # Resolução real (câmera pode arredondar para a mais próxima).
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


def main():
    global _running, _custom_labels, _infer_ms_acc, _infer_count_window
    args = parse_args()

    # --- Model type (auto-detect se não especificado) ---
    model_type_name = args.model_type
    if not model_type_name:
        model_type_name = _detect_model_type(args.model)
        if not model_type_name:
            print("[ERRO] Não foi possível inferir --model-type pelo nome do arquivo.\n"
                  "  Especifique manualmente com --model-type.")
            print("  Tipos disponíveis: " + ", ".join(
                t for t in dir(nn.ModelType) if not t.startswith("_")))
            sys.exit(1)
        print(f"ModelType auto     : {model_type_name}")

    if args.labels:
        _custom_labels = _load_labels(args.labels)
        print(f"Labels carregados  : {len(_custom_labels)} classes (--labels)")
    else:
        _custom_labels = _labels_from_factory(model_type_name)
        if _custom_labels:
            print(f"Labels carregados  : {len(_custom_labels)} classes (model_factory.json)")

    # --- Modelo ---
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
    print(f"Limiar de confiança: {detector.get_threshold():.2f}")

    # --- Modelo stage 1 (opcional — para pipelines dois estágios) ---
    # Dois estágios para: landmarks/keypoints e atributos faciais (CLS_ATTRIBUTE).
    stage1_detector = None
    mt_upper = model_type_name.upper()
    _needs_two_stage = ("KEYPOINT" in mt_upper or "LANDMARK" in mt_upper
                        or "CLS_ATTRIBUTE" in mt_upper)
    if args.stage1_model:
        if not _needs_two_stage:
            print(f"[AVISO] --stage1-model ignorado: {model_type_name} não é um "
                  f"modelo que precise de dois estágios (landmarks/keypoints/cls_attribute).")
        else:
            stage1_type = getattr(nn.ModelType, args.stage1_model_type, None)
            if stage1_type is None:
                print(f"[ERRO] Stage-1 ModelType desconhecido: {args.stage1_model_type}")
                sys.exit(1)
            print(f"Carregando stage-1 : {args.stage1_model}")
            print(f"Stage-1 ModelType  : {args.stage1_model_type}")
            stage1_detector = nn.get_model(stage1_type, args.stage1_model)
            stage1_detector.set_threshold(args.threshold)

    # --- Fonte de vídeo ---
    global _source_type
    cam, _source_type = _open_source(args)

    # --- Auto-ajuste de fps/gop para câmera USB ---
    # Câmeras USB no Duo entregam ~3-5 fps após conversão de cor.  Declarar
    # fps=15 (padrão) ao encoder VENC causa mismatch no rate controller CBR
    # e GOP excessivamente longo (keyframe a cada 5s a 3fps).
    # Ajustar automaticamente quando o usuário não definiu explicitamente.
    fps = args.fps
    bitrate = args.bitrate
    if _source_type == "usb" and fps > 10:
        fps = 3
        print(f"  [auto] FPS ajustado para {fps} (câmera USB geralmente entrega ≤5fps)")
    if _source_type == "usb" and args.bitrate >= 2048:
        bitrate = 1024
        print(f"  [auto] Bitrate ajustado para {bitrate}kbps (USB: NALUs menores → "
              f"compatível com VLC/UDP)")
    gop = args.gop if args.gop > 0 else max(1, fps)
    if _source_type == "usb" and args.gop <= 0:
        gop = max(1, fps)
        print(f"  [auto] GOP ajustado para {gop} (1 keyframe/s para câmera USB)")

    # --- Servidor RTSP ---
    print(f"\nIniciando servidor RTSP {args.width}x{args.height} "
          f"codec={args.codec} bitrate={bitrate}kbps fps={fps} "
          f"gop={gop} ({gop/fps:.1f}s) sessão={args.session} ...")
    rtsp = image.RTSPServer(
        args.width, args.height,
        chn=0,
        codec=args.codec,
        session_name=args.session,
        bitrate=bitrate,
        gop=gop,
        fps=fps,
    )
    print(f"  Stream disponível em: rtsp://<ip-do-dispositivo>:554/{args.session}")
    print(f"  VLC: vlc --rtsp-tcp rtsp://<ip>:554/{args.session}")

    # --- Thread de inferência / release ---
    global _persist_detections
    _persist_detections = args.persist_detections
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(detector, cam, args.threshold),
        kwargs={"stage1_detector": stage1_detector},
        daemon=True,
    )
    infer_thread.start()

    # --- Loop principal (câmera + RTSP à taxa máxima) ---
    is_keypoint_model = ("KEYPOINT" in model_type_name.upper()
                         or "POSE" in model_type_name.upper())
    limit      = args.frames if args.frames > 0 else None
    frame_idx  = 0
    det_total  = 0
    t_start    = time.time()
    t_report   = t_start
    skip_every = args.skip_every

    # Acumuladores de tempo por janela de reporte
    _cam_ms_acc  = 0.0
    _draw_ms_acc = 0.0
    _send_ms_acc = 0.0
    _acc_count   = 0

    print("\nTransmitindo... (Ctrl+C para parar)\n")

    try:
        while _running and (limit is None or frame_idx < limit):
            # Semáforo apenas para câmera VI (VB pool limitado).
            # VDEC e USB gerenciam própria memória; sem semáforo necessário.
            if _source_type == "vi" and not _vb_sem.acquire(timeout=0.5):
                continue   # timeout — verifica _running e tenta de novo
            t0 = time.time()
            frame = cam.read()
            cam_ms = (time.time() - t0) * 1000

            # Lê últimas detecções e desenha overlay
            with _det_lock:
                dets = list(_last_detections)
                det_frame = _last_detect_frame

            # Com persist_detections: aplica TTL de skip_every frames.
            # Se a última inferência positiva foi há >= skip_every frames, expira.
            if _persist_detections and (frame_idx - det_frame) >= skip_every:
                dets = []

            t1 = time.time()
            if dets:
                det_total += len(dets) if isinstance(dets, (list, tuple)) else 1
                _draw_inference(frame, dets, is_keypoint_model, args.threshold)
            draw_ms = (time.time() - t1) * 1000

            t2 = time.time()
            rtsp.send_frame(frame)
            send_ms = (time.time() - t2) * 1000

            # Para RtspClientVdec: pina o frame para a thread de inferência e
            # libera o slot de display antes do próximo read().
            if _source_type == "vdec":
                cam.pin_for_inference()

            # Todos os frames passam pela _release_queue.
            # A thread de inferência chama cam.release() em ordem FIFO para
            # todos eles, garantindo que o VB block do frame N nunca seja
            # libertado antes de o frame N-1 ter sido processado.
            # Se a fila já está cheia, desativa a inferência neste frame para
            # evitar acumulação de VB blocks (pool de 4 blocos típico).
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
            _frame_count = frame_idx

            _cam_ms_acc  += cam_ms
            _draw_ms_acc += draw_ms
            _send_ms_acc += send_ms
            _acc_count   += 1

            # Relatório a cada 5 s
            now = time.time()
            if now - t_report >= 5.0:
                elapsed  = now - t_start
                cam_fps  = frame_idx / elapsed if elapsed > 0 else 0
                n = max(_acc_count, 1)
                ni = max(_infer_count_window, 1)
                avg_infer = _infer_ms_acc / ni
                print(f"  frame {frame_idx:6d}  "
                      f"cam={cam_fps:5.1f}fps  "
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
                t_report = now

    except Exception as exc:
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        if _source_type == "vi":
            _vb_sem.release()   # desbloqueia acquire() caso esteja esperando
        infer_thread.join(timeout=2.0)
        # Descarta frames pendentes na fila; cam.close() libera o frameQueues
        # interno do ViDecoder, então não é preciso chamar cam.release() aqui.
        with _release_lock:
            _release_queue.clear()
        if stage1_detector is not None:
            try:
                stage1_detector.close()
            except Exception:
                pass
        detector.close()
        cam.close()
        del rtsp

    elapsed = time.time() - t_start
    fps     = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\nFinalizado.")
    print(f"  Frames transmitidos : {frame_idx}")
    print(f"  Tempo total         : {elapsed:.1f} s")
    print(f"  FPS médio câmera    : {fps:.1f}")
    print(f"  FPS inferência      : {_infer_fps:.1f}")
    print(f"  Total de detecções  : {det_total}")


if __name__ == "__main__":
    main()
