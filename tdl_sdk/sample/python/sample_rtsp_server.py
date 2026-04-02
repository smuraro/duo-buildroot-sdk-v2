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
        [--codec h264] \\
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
    p.add_argument("--codec", default="h264", choices=["h264", "h265"],
                   help="Codec de vídeo: h264 ou h265 (padrão: h264)")
    p.add_argument("--session", default="live",
                   help="Nome da sessão RTSP (padrão: live)")
    p.add_argument("--frames", type=int, default=0,
                   help="Número de frames a transmitir; 0 = infinito (padrão: 0)")
    return p.parse_args()


# ─── Thread de inferência ─────────────────────────────────────────────────────
#
# Consome frames da fila _infer_queue, corre a inferência e publica os
# resultados em _last_detections (protegido por _det_lock).

_infer_queue     = []          # lista usada como fila de 1 elemento
_infer_lock      = threading.Lock()
_last_detections = []
_det_lock        = threading.Lock()
_infer_fps       = 0.0


def _inference_worker(detector, threshold):
    global _running, _infer_fps
    t0      = time.time()
    count   = 0
    while _running:
        frame = None
        with _infer_lock:
            if _infer_queue:
                frame = _infer_queue.pop(0)

        if frame is None:
            time.sleep(0.001)
            continue

        dets = detector.inference(frame)
        with _det_lock:
            _last_detections[:] = dets

        count += 1
        elapsed = time.time() - t0
        if elapsed > 0:
            _infer_fps = count / elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _running
    args = parse_args()

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
          f"codec={args.codec} sessão={args.session} ...")
    rtsp = image.RTSPServer(
        args.width, args.height,
        chn=0,
        codec=args.codec,
        session_name=args.session,
    )
    print(f"  Stream disponível em: rtsp://<ip-do-dispositivo>:554/{args.session}")

    # --- Câmera ---
    print(f"\nAbrindo câmera {args.width}x{args.height} ...")
    cam = image.Camera(args.width, args.height, image.ImageFormat.YUV420SP_VU)

    # --- Thread de inferência ---
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(detector, args.threshold),
        daemon=True,
    )
    infer_thread.start()

    # --- Loop principal (câmera + RTSP à taxa máxima) ---
    limit      = args.frames if args.frames > 0 else None
    frame_idx  = 0
    det_total  = 0
    t_start    = time.time()
    t_report   = t_start
    skip_every = 2   # envia 1 em cada N frames para inferência

    print("\nTransmitindo... (Ctrl+C para parar)\n")

    try:
        while _running and (limit is None or frame_idx < limit):
            frame = cam.read()

            # Envia para inferência (sem bloquear): descarta se ainda ocupada
            if frame_idx % skip_every == 0:
                with _infer_lock:
                    _infer_queue.clear()
                    _infer_queue.append(frame)

            # Lê últimas detecções e desenha overlay
            with _det_lock:
                dets = list(_last_detections)

            if dets:
                det_total += len(dets)
                image.draw_detections(frame, dets, score_threshold=args.threshold)

            rtsp.send_frame(frame)
            cam.release()

            frame_idx += 1

            # Relatório a cada 5 s
            now = time.time()
            if now - t_report >= 5.0:
                elapsed  = now - t_start
                cam_fps  = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  frame {frame_idx:6d}  "
                      f"cam_fps={cam_fps:5.1f}  "
                      f"inf_fps={_infer_fps:4.1f}  "
                      f"dets={len(dets)}  "
                      f"total={det_total}")
                t_report = now

    except Exception as exc:
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        infer_thread.join(timeout=2.0)
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
