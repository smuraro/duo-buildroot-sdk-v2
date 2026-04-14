#!/usr/bin/env python3
"""
sample_rtsp_client.py — Lê frames de um stream RTSP e executa inferência

Suporta dois backends de decodificação:
  --backend opencv   (padrão) OpenCV/FFmpeg: simples, decode por software
  --backend vdec     live555 + VDEC hardware: zero CPU para decode H264
                     (streams H265 usam fallback OpenCV automaticamente)

Uso:
    python3 sample_rtsp_client.py \\
        --input  rtsp://192.168.1.10:554/live \\
        --model  /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        [--output-rtsp rtsp://0.0.0.0:554/out]
"""

import argparse
import sys
import time

import tdl
from tdl import image, nn


def _detect_model_type(model_path):
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


def _open_client(url, args):
    if args.backend == "vdec":
        try:
            client = image.RtspClientVdec(url,
                                          width=args.width, height=args.height,
                                          transport=args.transport)
            if client.is_opened():
                print("Backend: vdec (hardware H264 decode)")
                return client
            print("VDEC indisponível, usando fallback OpenCV")
        except Exception as e:
            print(f"VDEC falhou ({e}), usando fallback OpenCV")

    if not hasattr(image, "RtspClient"):
        print("Erro: backend 'opencv' não disponível nesta build. Use --backend vdec.")
        sys.exit(1)
    print("Backend: opencv (decode por software)")
    return image.RtspClient(url,
                            width=args.width, height=args.height,
                            transport=args.transport)


def main():
    p = argparse.ArgumentParser(description="RTSP client com inferência TDL")
    p.add_argument("--input",       required=True,
                   help="URL do stream RTSP de entrada")
    p.add_argument("--model",       required=True,
                   help="Caminho para o arquivo .cvimodel")
    p.add_argument("--model-type",  default="", dest="model_type",
                   help="Nome do ModelType (ex: SCRFD_DET_FACE, YOLOV8_DET_COCO80)")
    p.add_argument("--backend",     default="vdec", choices=["opencv", "vdec"],
                   help="Backend de decodificação")
    p.add_argument("--width",       type=int, default=0)
    p.add_argument("--height",      type=int, default=0)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--output-rtsp", default="", dest="output_rtsp",
                   help="URL do re-stream RTSP de saída com overlay "
                        "(ex: rtsp://0.0.0.0:554/out). Requer --width e --height.")
    p.add_argument("--codec",       default="h264", choices=["h264", "h265"])
    p.add_argument("--bitrate",     type=int, default=2048,
                   help="Bitrate do encoder em kbps (padrão=2048). "
                        "Reduza se a rede for lenta (WiFi: 512-1024).")
    p.add_argument("--gop",         type=int, default=0,
                   help="Intervalo de keyframe em frames (padrão=0=automático: 1× fps). "
                        "Valores menores → refresh mais frequente.")
    p.add_argument("--fps",         type=int, default=15,
                   help="Frame rate declarado ao encoder VENC (padrão=15). "
                        "Deve refletir o FPS real — valor errado causa qualidade ruim.")
    p.add_argument("--frames",      type=int, default=0,
                   help="Número de frames a processar; 0 = infinito")
    p.add_argument("--transport",   default="tcp", choices=["tcp", "udp"])
    p.add_argument("--infer-every", type=int, default=1, dest="infer_every",
                   help="Executar inferência a cada N frames (padrão=1). "
                        "Use 2+ para reduzir carga do NPU.")
    p.add_argument("--soft-nms",    action="store_true", default=False, dest="soft_nms")
    args = p.parse_args()

    # ── Modelo ────────────────────────────────────────────────────────────────
    model_type = args.model_type
    if not model_type:
        model_type = _detect_model_type(args.model)
        if not model_type:
            print("Erro: não foi possível inferir --model-type. Especifique manualmente.")
            sys.exit(1)

    model_type_enum = getattr(nn.ModelType, model_type, None)
    if model_type_enum is None:
        print(f"Erro: ModelType desconhecido: '{model_type}'")
        print("Tipos disponíveis: " + ", ".join(
            t for t in dir(nn.ModelType) if not t.startswith("_")))
        sys.exit(1)

    print(f"Carregando modelo: {args.model}  tipo={model_type}")
    model = nn.get_model(model_type_enum, args.model)
    if model is None:
        print(f"Erro: modelo '{model_type}' não reconhecido")
        sys.exit(1)
    model.set_threshold(args.threshold)
    if args.soft_nms:
        model.set_soft_nms(True)
        print("Soft NMS habilitado")

    is_kp = "KEYPOINT" in model_type.upper() or "POSE" in model_type.upper()
    limit = args.frames if args.frames > 0 else None

    # ── RTSP output ───────────────────────────────────────────────────────────
    # RTSPServer deve ser criado ANTES do RtspClientVdec (VDEC/Wave4 ordering).
    # GOP automático: 1 I-frame por segundo (mínimo 1).
    gop = args.gop if args.gop > 0 else max(1, args.fps)
    rtsp_out = None
    if args.output_rtsp:
        if args.width == 0 or args.height == 0:
            print("Erro: --output-rtsp requer --width e --height.")
            sys.exit(1)
        session = args.output_rtsp.rstrip("/").split("/")[-1] or "out"
        print(f"Encoder: codec={args.codec}  bitrate={args.bitrate}kbps  "
              f"fps={args.fps}  gop={gop} ({gop/args.fps:.1f}s)")
        print(f"Re-stream RTSP: rtsp://<ip>:554/{session}")
        rtsp_out = image.RTSPServer(args.width, args.height,
                                    chn=0,
                                    codec=args.codec,
                                    session_name=session,
                                    bitrate=args.bitrate,
                                    gop=gop,
                                    fps=args.fps)

    # ── Abre stream ───────────────────────────────────────────────────────────
    client = _open_client(args.input, args)
    print(f"Aberto: {args.input}  is_opened={client.is_opened()}\n")

    frame_idx = 0
    t_start   = time.time()
    last_dets = None

    t_read = t_infer = t_draw = t_venc = 0.0
    REPORT_INTERVAL = 30

    try:
        while limit is None or frame_idx < limit:
            t0 = time.perf_counter()
            try:
                frame = client.read()
            except RuntimeError as e:
                print(f"Fim do stream ou erro: {e}")
                break
            t1 = time.perf_counter()

            if frame_idx % args.infer_every == 0:
                last_dets = model.inference(frame)
            t2 = time.perf_counter()

            n = 0
            if last_dets:
                n = len(last_dets) if isinstance(last_dets, (list, tuple)) else 1
                if rtsp_out is not None:
                    if is_kp:
                        image.draw_keypoints(frame, last_dets, args.threshold)
                    else:
                        image.draw_detections(frame, last_dets, args.threshold)
            t3 = time.perf_counter()

            if rtsp_out is not None:
                rtsp_out.send_frame(frame)
            t4 = time.perf_counter()

            t_read  += t1 - t0
            t_infer += t2 - t1
            t_draw  += t3 - t2
            t_venc  += t4 - t3

            if frame_idx % REPORT_INTERVAL == 0:
                elapsed = time.time() - t_start
                fps_real = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                f = frame_idx + 1
                print(
                    f"frame {frame_idx:5d}  dets={n:<3d}  fps={fps_real:4.1f}"
                    f"  read={t_read/f*1000:5.1f}ms"
                    f"  infer={t_infer/f*1000:5.1f}ms"
                    f"  draw={t_draw/f*1000:4.1f}ms"
                    f"  venc={t_venc/f*1000:5.1f}ms"
                    f"  total={(t_read+t_infer+t_draw+t_venc)/f*1000:5.1f}ms"
                )

            client.release()
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        client.close()
        elapsed = time.time() - t_start
        if frame_idx > 0:
            f = frame_idx
            print(
                f"\nRESUMO: {frame_idx} frames  {elapsed:.1f}s  "
                f"{frame_idx/elapsed:.1f}fps\n"
                f"  read={t_read/f*1000:.1f}ms  infer={t_infer/f*1000:.1f}ms  "
                f"draw={t_draw/f*1000:.1f}ms  venc={t_venc/f*1000:.1f}ms  "
                f"total={(t_read+t_infer+t_draw+t_venc)/f*1000:.1f}ms/frame"
            )


if __name__ == "__main__":
    main()
