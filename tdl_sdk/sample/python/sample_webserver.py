#!/usr/bin/env python3
"""
sample_webserver.py — Câmera + inferência TDL + servidor web

Abre a câmera, carrega um modelo TDL, desenha as detecções em cada frame
e serve um preview JPEG anotado numa interface web na porta 9000.

Suporta três fontes de vídeo (--input):
  vazio         câmera VI local (padrão)
  rtsp://...    stream RTSP decodificado por hardware (VDEC)
  usb / usb:N   câmera USB /dev/video0 (ou /dev/videoN)

Uso (câmera VI):
    python3 sample_webserver.py \\
        --model /root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel \\
        [--model-type SCRFD_DET_FACE] [--width 1280] [--height 720]

Uso (entrada RTSP via VDEC):
    python3 sample_webserver.py \\
        --model ... --input rtsp://192.168.1.10:554/live [--transport tcp|udp]

Uso (câmera USB):
    python3 sample_webserver.py \\
        --model ... --input usb [--width 640] [--height 480]

Acesso:
    http://<ip-do-dispositivo>:9000/
"""

import argparse
import base64
import json
import os
import signal
import sys
import threading
import time

import tdl
from tdl import image, nn
from http.server import HTTPServer, BaseHTTPRequestHandler

# COCO80 class names (0-indexed, standard order)
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


# Labels customizados carregados em runtime (sobrepõem COCO80 e cls<N>)
_custom_labels: dict = {}


def _coco_name(class_id: int, class_name: str) -> str:
    """Return proper class name: custom labels → COCO80 → original."""
    if class_name.startswith("cls") or class_name == "UNDEFINED":
        if class_id in _custom_labels:
            return _custom_labels[class_id]
        if 0 <= class_id < len(COCO80_NAMES):
            return COCO80_NAMES[class_id]
    return class_name

# ─── Shared state ─────────────────────────────────────────────────────────────

_lock              = threading.Lock()
_latest_jpeg_b64   = ""
_latest_detections = []   # list of label strings, e.g. ["face 0.92", "face 0.81"]
_latest_perf       = [0, 0, 0]   # [inf_ms, draw_ms, jpeg_ms]
_frame_count       = 0
_status            = "Initializing..."
_running           = True


def _set_state(jpeg_b64=None, detections=None, perf=None, status=None,
               inc_frame=False):
    global _latest_jpeg_b64, _latest_detections, _latest_perf, \
           _frame_count, _status
    with _lock:
        if jpeg_b64    is not None: _latest_jpeg_b64   = jpeg_b64
        if detections  is not None: _latest_detections = detections
        if perf        is not None: _latest_perf       = perf
        if status      is not None: _status            = status
        if inc_frame:               _frame_count      += 1


def _get_state():
    with _lock:
        return (_latest_jpeg_b64, _latest_detections[:],
                _latest_perf[:], _frame_count, _status)


# ─── Inference loop ───────────────────────────────────────────────────────────

def _build_det_labels(detections):
    """Convert inference output to human-readable label strings."""
    labels = []
    for d in detections:
        cls = _coco_name(d.get("class_id", -1), d.get("class_name", "?"))
        score = d.get("score", 0.0)
        labels.append(f"{cls} {score:.0%}")
    return labels


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
                    dets = [{**d, "class_name": _coco_name(
                                d.get("class_id", -1),
                                f"cls{d.get('class_id', 0)}")}
                            for d in dets if isinstance(d, dict)]
                image.draw_classification(frame, dets)
                return
            # OBJECT_DETECTION / OBJECT_DETECTION_WITH_LANDMARKS
            enriched = []
            for d in dets:
                d2 = dict(d)
                d2["class_name"] = _coco_name(d.get("class_id", -1),
                                               d.get("class_name", "?"))
                enriched.append(d2)
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


def inference_loop(args):
    global _running

    # Load model
    model_type = getattr(nn.ModelType, args.model_type, None)
    if model_type is None:
        print(f"[ERRO] ModelType desconhecido: {args.model_type}")
        sys.exit(1)

    _set_state(status=f"Carregando modelo {args.model_type}...")
    print(f"Carregando modelo  : {args.model}")
    print(f"ModelType          : {args.model_type}")
    detector = nn.get_model(model_type, args.model)
    detector.set_threshold(args.threshold)
    print(f"Limiar             : {detector.get_threshold():.2f}")

    # Open camera / video source
    _set_state(status="Abrindo fonte de video...")
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
        print(f"\nAbrindo câmera USB /dev/video{device}  {args.width}x{args.height} ...")
        cam = image.UsbCamera(device, args.width, args.height)
        if not cam.is_opened():
            print(f"[ERRO] Não foi possível abrir /dev/video{device}.")
            sys.exit(1)
        print(f"  Backend: USB V4L2 /dev/video{device}")
    elif inp.startswith("rtsp://") or inp.startswith("rtsps://"):
        print(f"\nAbrindo stream RTSP {inp} ({args.transport}) ...")
        cam = image.RtspClientVdec(inp, width=args.width, height=args.height,
                                   transport=args.transport)
        if not cam.is_opened():
            print("[ERRO] Não foi possível abrir o stream RTSP.")
            sys.exit(1)
        print("  Backend: VDEC hardware (H264)")
    else:
        print(f"\nAbrindo câmera VI {args.width}x{args.height} ...")
        cam = image.Camera(args.width, args.height, image.ImageFormat.YUV420SP_VU,
                           mirror=args.mirror, flip=args.flip)
        print("  Backend: câmera VI local")

    _set_state(status="Running")
    print("Iniciando loop de inferência...\n")

    # Detect if model has keypoints (for choosing draw function)
    is_keypoint_model = "KEYPOINT" in args.model_type.upper() \
                        or "POSE" in args.model_type.upper()

    # JPEG is only encoded at web_fps rate — inference runs at full speed
    # between encodes.  The browser polls at 200 ms so >5 fps brings no benefit.
    jpeg_interval = 1.0 / max(1, args.web_fps)
    last_jpeg_time = 0.0

    t_start      = time.time()
    t_report     = t_start
    frame_idx    = 0
    jpeg_ms      = 0
    _frame_limit = args.frames if args.frames > 0 else None

    try:
        while _running and (_frame_limit is None or frame_idx < _frame_limit):
            frame = cam.read()

            # Inference
            t0 = time.time()
            detections = detector.inference(frame)
            t1 = time.time()
            inf_ms = int((t1 - t0) * 1000)

            # Draw
            if detections:
                _draw_inference(frame, detections, is_keypoint_model,
                                args.threshold)
            t2 = time.time()
            draw_ms = int((t2 - t1) * 1000)

            # Encode JPEG only when the web preview interval has elapsed.
            # Skipping this on most frames is the main speedup — cvtColor +
            # imencode on 1280×720 costs ~30-50 ms on an embedded CPU.
            labels = _build_det_labels(detections)
            if t2 - last_jpeg_time >= jpeg_interval:
                jpeg_bytes = image.frame_to_jpeg(
                    frame,
                    quality=args.jpeg_quality,
                    scale=args.preview_scale,
                )
                jpeg_ms = int((time.time() - t2) * 1000)
                jpeg_b64 = base64.b64encode(jpeg_bytes).decode()
                last_jpeg_time = t2
                _set_state(jpeg_b64=jpeg_b64, detections=labels,
                           perf=[inf_ms, draw_ms, jpeg_ms],
                           status="Running", inc_frame=True)
            else:
                # No encode this frame — just update detections/perf
                _set_state(detections=labels,
                           perf=[inf_ms, draw_ms, jpeg_ms],
                           status="Running", inc_frame=True)

            cam.release()
            frame_idx += 1

            # Console report every 5 s
            now = time.time()
            if now - t_report >= 5.0:
                elapsed = now - t_start
                fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  frame {frame_idx:6d}  fps={fps:5.1f}  "
                      f"det={len(detections)}  "
                      f"inf={inf_ms}ms draw={draw_ms}ms jpeg={jpeg_ms}ms")
                t_report = now

    except Exception as exc:
        _set_state(status=f"Erro: {exc}")
        print(f"\n[ERRO] {exc}")
    finally:
        _running = False
        _set_state(status="Parado")
        detector.close()
        cam.close()

    elapsed = time.time() - t_start
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\nFinalizado. {frame_idx} frames em {elapsed:.1f}s ({fps:.1f} fps)")


# ─── Web server ───────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vision Inference</title>
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

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.07) 2px,
      rgba(0,0,0,0.07) 4px
    );
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

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
  }

  header h1 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
  }

  header .sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--dim);
    margin-left: auto;
  }

  main {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 0;
    flex: 1;
  }

  /* camera feed */
  .feed-wrap {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid var(--border);
    min-height: 420px;
  }

  .feed-wrap img {
    max-width: 100%;
    max-height: calc(100vh - 120px);
    display: block;
    object-fit: contain;
  }

  .no-feed {
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
    animation: spin 0.9s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* corner brackets */
  .corner {
    position: absolute;
    width: 22px; height: 22px;
    border-color: var(--accent);
    border-style: solid;
    opacity: 0.6;
  }
  .corner.tl { top: 12px; left: 12px; border-width: 2px 0 0 2px; }
  .corner.tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
  .corner.bl { bottom: 12px; left: 12px; border-width: 0 0 2px 2px; }
  .corner.br { bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }

  /* side panel */
  .panel {
    display: flex;
    flex-direction: column;
    gap: 0;
    background: var(--panel);
    overflow-y: auto;
  }

  .section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }

  .section-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 14px;
  }

  /* status badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent2);
  }

  .status-badge.warn { color: var(--warn); }

  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
    animation: blink 1.4s ease-in-out infinite;
  }

  /* metrics */
  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  .metric-row:last-child { border-bottom: none; }

  .metric-label { color: var(--dim); font-family: var(--mono); font-size: 11px; }
  .metric-val   { font-family: var(--mono); font-weight: 600; color: var(--text); }
  .metric-val.accent { color: var(--accent); }

  /* perf bars */
  .perf-bar-wrap { margin-top: 4px; }
  .perf-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    margin-bottom: 3px;
  }
  .perf-bar-bg {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-bottom: 8px;
    overflow: hidden;
  }
  .perf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .perf-bar-fill.draw { background: var(--accent2); }
  .perf-bar-fill.jpeg { background: #ff9f1c; }

  /* detections list */
  .det-list { list-style: none; }
  .det-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
  }
  .det-item:last-child { border-bottom: none; }
  .det-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .no-det { font-family: var(--mono); font-size: 12px; color: var(--dim); }

  /* footer */
  footer {
    padding: 10px 28px;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    display: flex;
    gap: 24px;
    background: var(--panel);
  }

  @media (max-width: 700px) {
    main { grid-template-columns: 1fr; }
    .feed-wrap { border-right: none; border-bottom: 1px solid var(--border); }
    .panel { max-height: 340px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>Vision Inference</h1>
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
      <span id="status-text">Waiting for camera...</span>
    </div>
    <img id="feed-img" style="display:none" alt="Inference output">
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
        <span class="metric-label">FPS (live)</span>
        <span class="metric-val" id="fps-live">--</span>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Timing (ms)</div>
      <div class="perf-bar-wrap">
        <div class="perf-bar-label"><span>INF</span><span id="inf-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill" id="inf-bar" style="width:0%"></div></div>
        <div class="perf-bar-label"><span>DRAW</span><span id="draw-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill draw" id="draw-bar" style="width:0%"></div></div>
        <div class="perf-bar-label"><span>JPEG</span><span id="jpeg-val">0</span></div>
        <div class="perf-bar-bg"><div class="perf-bar-fill jpeg" id="jpeg-bar" style="width:0%"></div></div>
      </div>
    </div>

    <div class="section" style="flex:1">
      <div class="section-title">Detections</div>
      <ul class="det-list" id="det-list">
        <li class="no-det">No detections yet</li>
      </ul>
    </div>
  </aside>
</main>

<footer>
  <span>TDL DIRECT</span>
  <span id="footer-model">model: --</span>
  <span id="footer-time">--</span>
</footer>

<script>
const COLORS = [
  '#385e0f','#0080ff','#0000ff','#ff0000',
  '#00ff00','#ffff00','#00ffff','#ff00ff',
  '#8000ff','#ff8000'
];

let lastFrameCount = 0;
let lastFrameTime  = Date.now();
let fpsVal         = 0;

function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }
function barWidth(ms, maxMs=500){
  return clamp(ms/maxMs*100, 0, 100).toFixed(1)+'%';
}

async function poll(){
  try {
    const r = await fetch('/api/state');
    if(!r.ok) return;
    const d = await r.json();

    document.getElementById('hdr-model').textContent = d.model || '--';

    const badge = document.getElementById('status-badge');
    const badgeText = document.getElementById('badge-text');
    const running = d.status === 'Running';
    badge.className = 'status-badge' + (running ? '' : ' warn');
    badgeText.textContent = d.status || 'init';
    document.getElementById('status-text').textContent = d.status || '...';

    const fc = d.frame_count || 0;
    document.getElementById('frame-count').textContent = fc;

    const now = Date.now();
    const elapsed = (now - lastFrameTime)/1000;
    if(elapsed >= 1.0){
      fpsVal = ((fc - lastFrameCount)/elapsed).toFixed(1);
      lastFrameCount = fc;
      lastFrameTime  = now;
    }
    document.getElementById('fps-live').textContent = fpsVal;

    const img    = document.getElementById('feed-img');
    const noFeed = document.getElementById('no-feed');
    if(d.image){
      img.src = 'data:image/jpeg;base64,' + d.image;
      img.style.display = 'block';
      noFeed.style.display = 'none';
    } else {
      img.style.display = 'none';
      noFeed.style.display = 'flex';
    }

    const perf = d.perf || [0,0,0];
    document.getElementById('inf-val').textContent  = perf[0];
    document.getElementById('draw-val').textContent = perf[1];
    document.getElementById('jpeg-val').textContent = perf[2];
    document.getElementById('inf-bar').style.width  = barWidth(perf[0]);
    document.getElementById('draw-bar').style.width = barWidth(perf[1]);
    document.getElementById('jpeg-bar').style.width = barWidth(perf[2]);

    const list = document.getElementById('det-list');
    const dets = d.detections || [];
    if(dets.length === 0){
      list.innerHTML = '<li class="no-det">No detections</li>';
    } else {
      list.innerHTML = dets.map((lbl, i) =>
        `<li class="det-item">
          <span class="det-dot" style="background:${COLORS[i%COLORS.length]}"></span>
          <span>${lbl}</span>
        </li>`
      ).join('');
    }

    document.getElementById('footer-model').textContent = 'model: ' + (d.model||'--');
    document.getElementById('footer-time').textContent = new Date().toLocaleTimeString();

  } catch(e){ /* ignore */ }
}

setInterval(poll, 200);
poll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    model_name = ""

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
        elif self.path == "/api/state":
            jpeg_b64, dets, perf, fc, status = _get_state()
            payload = json.dumps({
                "image":       jpeg_b64,
                "detections":  dets,
                "perf":        perf,
                "frame_count": fc,
                "status":      status,
                "model":       Handler.model_name,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()


def start_web(port, model_name):
    Handler.model_name = model_name
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Web UI disponível em http://0.0.0.0:{port}")
    server.serve_forever()


# ─── Signal handling ──────────────────────────────────────────────────────────

def _sigint_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Câmera + inferência TDL + servidor web")
    p.add_argument("--model",         required=True,
                   help="Caminho para o arquivo .cvimodel")
    p.add_argument("--model-type",    default="SCRFD_DET_FACE",
                   help="Nome do ModelType (padrão: SCRFD_DET_FACE)")
    p.add_argument("--width",         type=int, default=1280,
                   help="Largura da câmera em pixels (padrão: 1280)")
    p.add_argument("--height",        type=int, default=720,
                   help="Altura da câmera em pixels (padrão: 720)")
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Limiar de confiança (padrão: 0.5)")
    p.add_argument("--jpeg-quality",   type=int,   default=40,  dest="jpeg_quality",
                   help="Qualidade JPEG do preview web 0-100 (padrão: 40). "
                        "Com scale=1.0 a codificação é feita em hardware; "
                        "qualidade 40 resulta em tamanho similar ao scale=0.5/quality=75.")
    p.add_argument("--preview-scale", type=float, default=1.0, dest="preview_scale",
                   help="Fator de escala do preview JPEG (padrão: 1.0 = resolução completa "
                        "com codificação hardware). Use < 1.0 para forçar caminho software "
                        "com resolução reduzida (ex: 0.5 = metade).")
    p.add_argument("--web-fps",        type=int,   default=5,   dest="web_fps",
                   help="Máximo de frames JPEG encodados por segundo para o web preview "
                        "(padrão: 5). Inferência roda mais rápido entre os encodes.")
    p.add_argument("--web-port",       type=int,   default=9000, dest="web_port",
                   help="Porta do servidor web (padrão: 9000)")
    p.add_argument("--input",     default="",
                   help="Fonte de vídeo: vazio = câmera VI local; "
                        "rtsp://... = stream RTSP (VDEC hardware); "
                        "usb = /dev/video0; usb:1 = /dev/video1.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "udp"],
                   help="Transporte RTSP de entrada (padrão: tcp)")
    p.add_argument("--mirror", action="store_true", default=False,
                   help="Espelhar horizontalmente a imagem da câmera. Apenas câmera VI.")
    p.add_argument("--flip",  action="store_true", default=False,
                   help="Inverter verticalmente a imagem da câmera. Apenas câmera VI.")
    p.add_argument("--labels", default="", dest="labels",
                   help="Nomes das classes para modelos genéricos (YOLOV26, YOLOV8…). "
                        "Aceita caminho para arquivo .txt (uma classe por linha) "
                        "ou lista separada por vírgula: 'gato,cachorro,pássaro'.")
    p.add_argument("--frames",  type=int, default=0,
                   help="Número de frames a processar; 0 = infinito (padrão)")
    return p.parse_args()


def main():
    global _custom_labels
    args = parse_args()
    if args.labels:
        _custom_labels = _load_labels(args.labels)
        print(f"Labels carregados  : {len(_custom_labels)} classes (--labels)")
    else:
        _custom_labels = _labels_from_factory(args.model_type)
        if _custom_labels:
            print(f"Labels carregados  : {len(_custom_labels)} classes (model_factory.json)")
    model_name = os.path.basename(args.model)

    # Start web server in a background daemon thread
    t = threading.Thread(
        target=start_web,
        args=(args.web_port, model_name),
        daemon=True,
    )
    t.start()

    # Run inference loop in the main thread (blocks until SIGINT/SIGTERM)
    inference_loop(args)


if __name__ == "__main__":
    main()
