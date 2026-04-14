#!/usr/bin/env python3
"""
sample_rtsp_server.py — Câmera + detecção de objetos + servidor RTSP

Captura frames continuamente numa thread dedicada e transmite ao cliente
RTSP à taxa máxima da câmera. A inferência corre em paralelo; o overlay de
detecções é actualizado a cada novo resultado sem bloquear o stream.

Uso:
    python3 sample_rtsp_server.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        [--model-type SCRFD_DET_FACE] \\
        [--width 1280] [--height 720] \\
        [--threshold 0.5] \\
        [--codec h264] [--bitrate 3072] [--gop 15] [--skip-every 1] [--persist-detections] \\
        [--session live] \\
        [--frames 0]

Conectar ao stream:
    vlc rtsp://<ip-do-dispositivo>:554/<session>
    ffplay rtsp://<ip-do-dispositivo>:554/<session>

Parâmetros:
    --model       Caminho para o arquivo .cvimodel  (obrigatório)
    --model-type  Nome do ModelType (padrão: SCRFD_DET_FACE)
    --width       Largura da câmera em pixels (padrão: 1280)
    --height      Altura da câmera em pixels  (padrão: 720)
    --threshold   Limiar de confiança para detecções (padrão: 0.5)
    --codec       Codec de vídeo: h264 ou h265 (padrão: h264)
    --session     Nome da sessão RTSP / caminho da URL (padrão: live)
    --frames      Número de frames a transmitir; 0 = infinito (padrão: 0)
"""

import argparse
import signal
import sys
import threading
import time

import tdl
from tdl import image, nn


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
    p.add_argument("--model-type", default="SCRFD_DET_FACE",
                   help="Nome do ModelType (padrão: SCRFD_DET_FACE)")
    p.add_argument("--width",  type=int, default=1280,
                   help="Largura da câmera em pixels (padrão: 1280)")
    p.add_argument("--height", type=int, default=720,
                   help="Altura da câmera em pixels (padrão: 720)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Limiar de confiança (padrão: 0.5)")
    p.add_argument("--bitrate",   type=int, default=3072,
                   help="Bitrate de codificação em kbps (padrão: 3072). "
                        "Aumente para melhor qualidade em movimento (ex: 4096, 6144).")
    p.add_argument("--gop",       type=int, default=15,
                   help="Intervalo de keyframe em frames (padrão: 15). "
                        "Menor = melhor qualidade em movimento; maior = melhor compressão estática.")
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
    p.add_argument("--mirror", action="store_true", default=False,
                   help="Espelhar horizontalmente a imagem da câmera (flip esquerda↔direita).")
    p.add_argument("--flip",  action="store_true", default=False,
                   help="Inverter verticalmente a imagem da câmera (flip cima↔baixo).")
    p.add_argument("--labels", default="", dest="labels",
                   help="Nomes das classes para modelos genéricos (YOLOV26, YOLOV8…). "
                        "Aceita caminho para arquivo .txt (uma classe por linha) "
                        "ou lista separada por vírgula: 'gato,cachorro,pássaro'.")
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


def _inference_worker(detector, cam, threshold):
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
            ti   = time.time()
            dets = detector.inference(frame)
            _infer_ms = (time.time() - ti) * 1000

            with _det_lock:
                _last_detections[:] = dets
                if dets:
                    _last_detect_frame = _frame_count

            count += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                _infer_fps = count / elapsed

        # Sempre libera em ordem FIFO — nunca o main thread chama cam.release()
        cam.release()
        _vb_sem.release()   # libera um slot do pool para o main thread


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _running, _custom_labels
    args = parse_args()

    if args.labels:
        _custom_labels = _load_labels(args.labels)
        print(f"Labels carregados  : {len(_custom_labels)} classes (--labels)")
    else:
        _custom_labels = _labels_from_factory(args.model_type)
        if _custom_labels:
            print(f"Labels carregados  : {len(_custom_labels)} classes (model_factory.json)")

    # --- Modelo ---
    model_type = getattr(nn.ModelType, args.model_type, None)
    if model_type is None:
        print(f"[ERRO] ModelType desconhecido: {args.model_type}")
        sys.exit(1)

    print(f"Carregando modelo  : {args.model}")
    print(f"ModelType          : {args.model_type}")
    detector = nn.get_model(model_type, args.model)
    detector.set_threshold(args.threshold)
    print(f"Limiar de confiança: {detector.get_threshold():.2f}")

    # --- Servidor RTSP ---
    print(f"\nIniciando servidor RTSP {args.width}x{args.height} "
          f"codec={args.codec} bitrate={args.bitrate}kbps gop={args.gop} sessão={args.session} ...")
    rtsp = image.RTSPServer(
        args.width, args.height,
        chn=0,
        codec=args.codec,
        session_name=args.session,
        bitrate=args.bitrate,
        gop=args.gop,
    )
    print(f"  Stream disponível em: rtsp://<ip-do-dispositivo>:554/{args.session}")

    # --- Câmera ---
    print(f"\nAbrindo câmera {args.width}x{args.height} ...")
    cam = image.Camera(args.width, args.height, image.ImageFormat.YUV420SP_VU,
                       vb_buffer_num=_VB_BUFFER_NUM,
                       mirror=args.mirror, flip=args.flip)

    # --- Thread de inferência / release ---
    global _persist_detections
    _persist_detections = args.persist_detections
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(detector, cam, args.threshold),
        daemon=True,
    )
    infer_thread.start()

    # --- Loop principal (câmera + RTSP à taxa máxima) ---
    is_keypoint_model = ("KEYPOINT" in args.model_type.upper()
                         or "POSE" in args.model_type.upper())
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
            # Aguarda um slot livre no pool de VB blocks antes de ler o próximo
            # frame.  Sem isso, se a inferência demorar mais que um ciclo de
            # câmera, o pool esgota-se e o ISP para de produzir frames.
            if not _vb_sem.acquire(timeout=0.5):
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
                _release_queue.append((frame, do_infer))
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
                print(f"  frame {frame_idx:6d}  "
                      f"cam={cam_fps:5.1f}fps  "
                      f"inf={_infer_fps:4.1f}fps  "
                      f"dets={len(dets)}  "
                      f"| read={_cam_ms_acc/n:5.1f}ms  "
                      f"inf={_infer_ms:5.1f}ms  "
                      f"draw={_draw_ms_acc/n:4.1f}ms  "
                      f"send={_send_ms_acc/n:5.1f}ms")
                _cam_ms_acc = _draw_ms_acc = _send_ms_acc = 0.0
                _acc_count  = 0
                t_report = now

    except Exception as exc:
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        _vb_sem.release()   # desbloqueia acquire() caso esteja esperando
        infer_thread.join(timeout=2.0)
        # Descarta frames pendentes na fila; cam.close() libera o frameQueues
        # interno do ViDecoder, então não é preciso chamar cam.release() aqui.
        with _release_lock:
            _release_queue.clear()
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
