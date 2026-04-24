#!/usr/bin/env python3
"""
test_usb_rtsp.py — Teste mínimo: câmera USB → RTSP com frame padding.

Diagnóstico: isola o problema de RTSP congelado com câmera USB.
Lê frames da câmera USB numa thread e envia ao RTSP server numa thread
separada a uma taxa constante (--fps), repetindo o último frame quando
não há frame novo disponível.

Uso:
    python3 test_usb_rtsp.py [--width 640] [--height 480] [--fps 15]

Conectar:
    vlc rtsp://<ip>:554/live
    ffplay -rtsp_transport tcp rtsp://<ip>:554/live
"""

import argparse
import signal
import sys
import threading
import time

import tdl
from tdl import image

_running = True

def _sighandler(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGINT, _sighandler)
signal.signal(signal.SIGTERM, _sighandler)


def main():
    p = argparse.ArgumentParser(description="Teste USB → RTSP com frame padding")
    p.add_argument("--device", type=int, default=0, help="USB device index")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=15,
                   help="Taxa de envio ao RTSP (padrão: 15)")
    p.add_argument("--codec",  default="h264", choices=["h264", "h265"])
    p.add_argument("--session", default="live")
    args = p.parse_args()

    # --- RTSP Server ---
    print(f"Criando RTSP server {args.width}x{args.height} fps={args.fps} ...")
    rtsp = image.RTSPServer(
        args.width, args.height,
        chn=0,
        codec=args.codec,
        session_name=args.session,
        fps=args.fps,
        gop=args.fps,  # 1 I-frame por segundo
    )
    print(f"  Stream: rtsp://<ip>:554/{args.session}")

    # --- USB Camera ---
    print(f"Abrindo câmera USB /dev/video{args.device} {args.width}x{args.height} ...")
    cam = image.UsbCamera(args.device, args.width, args.height)
    if not cam.is_opened():
        print("[ERRO] Câmera USB não abriu.")
        sys.exit(1)

    # --- Shared state ---
    _frame_lock = threading.Lock()
    _latest_frame = [None]  # mutable container for sharing between threads

    # --- Camera reader thread ---
    _cam_count = [0]
    def camera_reader():
        while _running:
            try:
                frame = cam.read()
                with _frame_lock:
                    _latest_frame[0] = frame
                _cam_count[0] += 1
            except Exception as e:
                print(f"[cam] erro: {e}")
                time.sleep(0.1)

    cam_thread = threading.Thread(target=camera_reader, daemon=True)
    cam_thread.start()

    # --- Aguardar primeiro frame ---
    print("Aguardando primeiro frame da câmera ...")
    while _running:
        with _frame_lock:
            if _latest_frame[0] is not None:
                break
        time.sleep(0.01)
    if not _running:
        return
    print("  Primeiro frame recebido!")

    # --- RTSP sender loop (main thread, taxa constante) ---
    interval = 1.0 / args.fps
    send_count = 0
    t_start = time.time()
    t_report = t_start
    last_cam_count = 0

    print(f"\nEnviando ao RTSP a {args.fps}fps (Ctrl+C para parar)\n")

    try:
        while _running:
            t0 = time.time()

            with _frame_lock:
                frame = _latest_frame[0]

            if frame is not None:
                rtsp.send_frame(frame)
                send_count += 1

            # Relatório a cada 5s
            now = time.time()
            if now - t_report >= 5.0:
                elapsed = now - t_start
                send_fps = send_count / elapsed if elapsed > 0 else 0
                cam_fps = _cam_count[0] / elapsed if elapsed > 0 else 0
                new_frames = _cam_count[0] - last_cam_count
                last_cam_count = _cam_count[0]
                print(f"  send={send_fps:.1f}fps  cam={cam_fps:.1f}fps  "
                      f"sent={send_count}  cam_frames={_cam_count[0]}  "
                      f"new_in_period={new_frames}")
                t_report = now

            # Manter taxa constante
            dt = time.time() - t0
            sleep_time = interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n[ERRO] {e}")
    finally:
        _running = False
        cam_thread.join(timeout=2.0)
        cam.close()
        del rtsp

    elapsed = time.time() - t_start
    print(f"\nFinalizado. {send_count} frames enviados em {elapsed:.1f}s "
          f"({send_count/elapsed:.1f} fps)")


if __name__ == "__main__":
    main()
