#!/usr/bin/env python3
"""
sample_rtsp_client.py — Lê frames de uma fonte de vídeo, executa inferência
e retransmite via RTSP com overlay de detecções.

Suporta três fontes de entrada (--input):
  rtsp://...   stream RTSP (backends: opencv ou vdec)
  usb          câmera USB /dev/video0
  usb:N        câmera USB /dev/videoN

Uso (câmera USB — mais simples):
    python3 sample_rtsp_client.py \\
        --input usb \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel

Uso (entrada RTSP — resolução auto-detectada):
    python3 sample_rtsp_client.py \\
        --input rtsp://192.168.1.10:554/live \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel

Uso (entrada RTSP — resolução manual):
    python3 sample_rtsp_client.py \\
        --input rtsp://192.168.1.10:554/live \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        --width 1280 --height 720

Conectar ao stream de saída:
    vlc --rtsp-tcp rtsp://<ip>:554/live
    ffplay rtsp://<ip>:554/live
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


def _probe_rtsp_resolution(url, transport="tcp"):
    """Conecta brevemente ao stream RTSP via OpenCV para descobrir a resolução.

    Retorna (width, height) ou (0, 0) se não conseguir.
    """
    try:
        import cv2
    except ImportError:
        return 0, 0
    try:
        transport_flag = ("tcp" if transport == "tcp" else "udp")
        # Configurar OpenCV para usar TCP (mais confiável para probe)
        import os
        env_key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        old = os.environ.get(env_key, "")
        os.environ[env_key] = f"rtsp_transport;{transport_flag}"
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


def _open_usb(inp, args):
    """Abre câmera USB e retorna (client, width, height)."""
    device = 0
    if ":" in inp:
        try:
            device = int(inp.split(":", 1)[1])
        except ValueError:
            pass
    if not hasattr(image, "UsbCamera"):
        print("Erro: UsbCamera não disponível nesta build (requer OpenCV videoio).")
        sys.exit(1)
    req_w = args.width if args.width else 640
    req_h = args.height if args.height else 480
    client = image.UsbCamera(device, req_w, req_h)
    if not client.is_opened():
        print(f"Erro: não foi possível abrir /dev/video{device}.")
        sys.exit(1)
    # Resolução real (câmera pode arredondar para a mais próxima).
    if hasattr(client, 'width') and hasattr(client, 'height'):
        w, h = client.width, client.height
    else:
        probe = client.read()
        w, h = probe.get_size()
        client.release()
    print(f"Backend: USB V4L2 /dev/video{device}  {w}x{h}")
    return client, w, h


def _open_rtsp(inp, args, width, height):
    """Abre stream RTSP e retorna (client, width, height)."""
    if args.backend == "vdec":
        try:
            client = image.RtspClientVdec(inp, width=width, height=height,
                                          transport=args.transport)
            if client.is_opened():
                print(f"Backend: vdec (hardware H264 decode)  {width}x{height}")
                return client, width, height
            print("VDEC indisponível, usando fallback OpenCV")
        except Exception as e:
            print(f"VDEC falhou ({e}), usando fallback OpenCV")

    if not hasattr(image, "RtspClient"):
        print("Erro: backend 'opencv' não disponível nesta build. Use --backend vdec.")
        sys.exit(1)
    print(f"Backend: opencv (decode por software)  {width}x{height}")
    return image.RtspClient(inp, width=width, height=height,
                            transport=args.transport), width, height


def main():
    p = argparse.ArgumentParser(description="Cliente de vídeo com inferência TDL")
    p.add_argument("--input",       required=True,
                   help="Fonte de vídeo: rtsp://... para RTSP; "
                        "usb para /dev/video0; usb:N para /dev/videoN")
    p.add_argument("--model",       required=True,
                   help="Caminho para o arquivo .cvimodel")
    p.add_argument("--model-type",  default="", dest="model_type",
                   help="Nome do ModelType (ex: SCRFD_DET_FACE, YOLOV8_DET_COCO80). "
                        "Auto-detectado pelo nome do arquivo se omitido.")
    p.add_argument("--backend",     default="vdec", choices=["opencv", "vdec"],
                   help="Backend de decodificação RTSP (padrão: vdec)")
    p.add_argument("--width",       type=int, default=0,
                   help="Largura (auto-detectado se omitido)")
    p.add_argument("--height",      type=int, default=0,
                   help="Altura (auto-detectado se omitido)")
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--session",     default="live",
                   help="Nome da sessão RTSP de saída (padrão: live)")
    p.add_argument("--no-rtsp",     action="store_true", default=False, dest="no_rtsp",
                   help="Desabilitar saída RTSP (só inferência, sem re-stream)")
    p.add_argument("--codec",       default="h264", choices=["h264", "h265"])
    p.add_argument("--bitrate",     type=int, default=2048,
                   help="Bitrate do encoder em kbps (padrão=2048)")
    p.add_argument("--gop",         type=int, default=0,
                   help="Intervalo de keyframe (padrão=0=automático: 1× fps)")
    p.add_argument("--fps",         type=int, default=15,
                   help="Frame rate declarado ao encoder VENC (padrão=15)")
    p.add_argument("--frames",      type=int, default=0,
                   help="Número de frames a processar; 0 = infinito")
    p.add_argument("--transport",   default="tcp", choices=["tcp", "udp"])
    p.add_argument("--infer-every", type=int, default=1, dest="infer_every",
                   help="Executar inferência a cada N frames (padrão=1)")
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

    # ── Abre fonte de vídeo ───────────────────────────────────────────────────
    is_usb = args.input.strip().lower().startswith("usb")

    if is_usb:
        client, frame_w, frame_h = _open_usb(args.input, args)
    else:
        # Resolução: usa --width/--height se fornecidos, senão auto-detecta.
        frame_w, frame_h = args.width, args.height
        if frame_w == 0 or frame_h == 0:
            print(f"Auto-detectando resolução de {args.input} ...")
            pw, ph = _probe_rtsp_resolution(args.input, args.transport)
            if pw > 0 and ph > 0:
                frame_w, frame_h = pw, ph
                print(f"  Resolução detectada: {frame_w}x{frame_h}")
            else:
                print("Erro: não foi possível detectar resolução do stream RTSP.\n"
                      "  Especifique manualmente com --width e --height.")
                sys.exit(1)
        client, frame_w, frame_h = _open_rtsp(args.input, args, frame_w, frame_h)

    # ── RTSP output ───────────────────────────────────────────────────────────
    fps = args.fps
    bitrate = args.bitrate
    if is_usb and fps > 10:
        fps = 3
        print(f"  [auto] FPS ajustado para {fps} (câmera USB geralmente entrega ≤5fps)")
    if is_usb and args.bitrate >= 2048:
        bitrate = 1024
        print(f"  [auto] Bitrate ajustado para {bitrate}kbps (USB: NALUs menores)")
    gop = args.gop if args.gop > 0 else max(1, fps)
    if is_usb and args.gop <= 0:
        gop = max(1, fps)
        print(f"  [auto] GOP ajustado para {gop} (1 keyframe/s)")

    rtsp_out = None
    if not args.no_rtsp:
        session = args.session
        print(f"Encoder: codec={args.codec}  bitrate={bitrate}kbps  "
              f"fps={fps}  gop={gop} ({gop/fps:.1f}s)")
        print(f"Re-stream RTSP: rtsp://<ip>:554/{session}")
        print(f"  VLC: vlc --rtsp-tcp rtsp://<ip>:554/{session}")
        rtsp_out = image.RTSPServer(frame_w, frame_h,
                                    chn=0,
                                    codec=args.codec,
                                    session_name=session,
                                    bitrate=bitrate,
                                    gop=gop,
                                    fps=fps)

    print(f"Aberto: {args.input}  {frame_w}x{frame_h}  is_opened={client.is_opened()}\n")

    # ── Loop principal ────────────────────────────────────────────────────────
    frame_idx = 0
    t_start   = time.time()
    last_dets = None

    t_read = t_infer = t_draw = t_venc = 0.0
    infer_count = 0
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
                t_infer += model.get_last_inference_ms() / 1000.0
                infer_count += 1
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
            t_draw  += t3 - t2
            t_venc  += t4 - t3

            if frame_idx % REPORT_INTERVAL == 0:
                elapsed = time.time() - t_start
                fps_real = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                f = frame_idx + 1
                ic = max(infer_count, 1)
                print(
                    f"frame {frame_idx:5d}  dets={n:<3d}  fps={fps_real:4.1f}"
                    f"  read={t_read/f*1000:5.1f}ms"
                    f"  infer={t_infer/ic*1000:5.1f}ms"
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
            ic = max(infer_count, 1)
            print(
                f"\nRESUMO: {frame_idx} frames  {elapsed:.1f}s  "
                f"{frame_idx/elapsed:.1f}fps\n"
                f"  read={t_read/f*1000:.1f}ms  infer={t_infer/ic*1000:.1f}ms  "
                f"draw={t_draw/f*1000:.1f}ms  venc={t_venc/f*1000:.1f}ms  "
                f"total={(t_read+t_infer+t_draw+t_venc)/f*1000:.1f}ms/frame"
            )


if __name__ == "__main__":
    main()
