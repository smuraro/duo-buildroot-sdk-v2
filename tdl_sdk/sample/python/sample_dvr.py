#!/usr/bin/env python3
"""
sample_dvr.py — DVR: câmera VI → H.264 → segmentos MP4 no SD card + web UI

Grava a câmera local (VI) em segmentos MP4 de duração configurável
(padrão: 30 s) no cartão SD e serve uma página web com player e uma barra
de thumbnails para navegar pelos arquivos já gravados.

Uso típico:
    python3 sample_dvr.py
    # abre http://<ip-do-dispositivo>:9001

Opções mais comuns:
    --out-dir /mnt/sd/dvr         Diretório de saída no SD
    --segment-seconds 30          Duração de cada segmento
    --width 1280 --height 720     Resolução da câmera
    --fps 15                      Frame rate (deve casar com o real)
    --bitrate 3072                Bitrate do encoder em kbps
    --gop 15                      Intervalo de keyframe em frames
    --web-port 9001               Porta do servidor web

Arquivos gerados:
    <out-dir>/dvr_YYYY-MM-DD_HH-MM-SS.mp4    vídeo do segmento
    <out-dir>/dvr_YYYY-MM-DD_HH-MM-SS.jpg    thumbnail do primeiro frame
"""

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import tdl
from tdl import image

import video_inference

# Globals that the web server reads.
_running    = True
_out_dir    = "/mnt/sd/dvr"
_stats_lock = threading.Lock()
_stats = {
    "frame_count":      0,
    "fps":              0.0,
    "started_ms":       int(time.time() * 1000),
    "current_segment":  "",
    "status":           "init",
}

# Live view: JPEG preview only encoded while at least one client is watching.
_jpeg_lock     = threading.Lock()
_latest_jpeg   = b""
_live_watchers = 0

# Pause/resume: record_loop finalizes the current segment when disabled and
# creates a fresh recorder on re-enable (each pause produces a new segment).
_recording_enabled = True

# Inference: engine mutates under _engine_lock; record_loop consults it via a
# local snapshot at the top of each iteration.
_engine_lock: threading.Lock = threading.Lock()
_engine:      "video_inference.VideoInferenceEngine | None" = None
_engine_name: str = ""  # current model name, "" = disabled
_engine_error: str = ""  # last load error, surfaced via /api/models

# SSE subscribers notified on each inference update.
_sse_cond    = threading.Condition()
_sse_version = 0  # bump every time _engine publishes new detections

# Args parsed at startup (read by handler threads for model switching).
_args = None


_sigint_count = 0
_shutdown_watchdog_started = False

def _shutdown_watchdog(timeout_s: float = 5.0):
    """Hard-exit se o shutdown gracioso não terminar em N segundos.
    Necessário porque rec.close()/cam.close() são C++ blocking: o handler
    Python do segundo SIGINT não roda até eles retornarem.

    OBS: se o driver de vídeo deixou o processo em estado D (uninterruptible),
    nem os._exit/SIGKILL matam — só reboot. A defesa nesse caso é evitar
    entrar nesse estado (ver _safe_shutdown_recorder)."""
    time.sleep(timeout_s)
    print(f"\n[dvr] shutdown timeout ({timeout_s}s) — forçando exit.",
          flush=True)
    os._exit(130)


def _sigint_handler(sig, frame):
    global _running, _sigint_count, _shutdown_watchdog_started
    _sigint_count += 1
    _running = False
    if _sigint_count >= 2:
        # Segundo Ctrl+C manual → exit imediato.
        print("\n[dvr] segundo SIGINT — forçando exit.", flush=True)
        os._exit(130)
    if not _shutdown_watchdog_started:
        _shutdown_watchdog_started = True
        threading.Thread(target=_shutdown_watchdog, daemon=True).start()


signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# ─── Recording loop ──────────────────────────────────────────────────────────

def _set_engine(model_name: str, model_type: str, args) -> dict:
    """Carrega (ou troca, ou desliga) o engine de inferência. Retorna um
    dict pronto pra JSON com {ok, name, error}."""
    global _engine, _engine_name, _engine_error
    with _engine_lock:
        # Fecha o atual
        if _engine is not None:
            try:
                _engine.close()
            except Exception:
                pass
            _engine = None
            _engine_name = ""
        # Nome vazio = desligar
        if not model_name:
            _engine_error = ""
            return {"ok": True, "name": ""}
        # Carrega novo
        try:
            # Resolve: aceita nome do enum OU path .cvimodel. Em ambos os
            # casos `resolved_name` fica sendo o nome canônico (enum key),
            # que é o que queremos guardar em _engine_name e expor ao UI.
            mt = video_inference.resolve_model_type(
                model_name, model_type or None)
            resolved_name = (model_type
                             or video_inference._name_from_path(model_name)
                             or model_name)
            eng = video_inference.VideoInferenceEngine(
                mt, model_dir=args.model_dir,
                threshold=args.model_threshold,
                model_name=resolved_name)
            _engine = eng
            _engine_name = resolved_name
            _engine_error = ""
            print(f"[dvr] modelo carregado: {resolved_name}")
            return {"ok": True, "name": resolved_name}
        except Exception as e:
            err = str(e)
            _engine_error = err
            print(f"\n[dvr] *** FALHA AO CARREGAR MODELO '{model_name}' ***",
                  flush=True)
            print(f"[dvr] erro: {err}\n", flush=True)
            return {"ok": False, "name": "", "error": err}


def _publish_detections() -> None:
    """Bumpa version e acorda subscribers SSE."""
    global _sse_version
    with _sse_cond:
        _sse_version += 1
        _sse_cond.notify_all()


def _make_recorder(args):
    return image.VideoRecorder(
        width=args.width, height=args.height,
        out_dir=args.out_dir,
        codec=args.codec,
        segment_seconds=args.segment_seconds,
        fps=args.fps,
        bitrate=args.bitrate,
        gop=args.gop,
    )


def record_loop(args):
    global _running, _latest_jpeg

    print(f"\nAbrindo câmera VI {args.width}x{args.height} @ {args.fps} fps ...")
    # vb_buffer_num=5: dá folga na FIFO VI→VPSS quando o record_loop
    # sofre jitter (inferência, flush de SD, web threads). Evita erros
    # "CSIBDG fifo overflow" em rajadas temporárias. Custo: ~5 buffers
    # a mais ocupados (cada ~1.4MB @ 1280x720 NV21 = ~7MB extra).
    cam = image.Camera(args.width, args.height,
                       image.ImageFormat.YUV420SP_VU,
                       vb_buffer_num=5,
                       mirror=args.mirror, flip=args.flip)

    print(f"Iniciando recorder {args.codec.upper()} "
          f"{args.bitrate}kbps seg={args.segment_seconds}s dir={args.out_dir}")
    rec = _make_recorder(args)

    with _stats_lock:
        _stats["status"] = "recording"

    t_start   = time.time()
    t_report  = t_start
    # send_frame must be paced to ~fps so the MP4 timestamps line up with
    # wall-clock duration.  We sleep between frames to cap the rate.
    period    = 1.0 / max(1, args.fps)
    next_tick = t_start
    frame_idx = 0
    # Live preview encode pace — keep much lower than capture fps to save CPU.
    jpeg_period    = 1.0 / max(1, args.live_fps)
    last_jpeg_time = 0.0
    # Track da rotação de segmento pra atualizar o manifest.js (consumido
    # pelo dvr_viewer.html offline).
    last_segment  = ""
    # Setado pelo handler de erro do send_frame — se True, o destrutor do
    # VENC trava no driver. Pulamos rec.close() nesse caso.
    _recorder_broken = False
    # [PROFILE] acumuladores zerados a cada report (5s)
    _prof_n = 0
    _prof_read = _prof_send = _prof_ai = _prof_jpeg = 0.0
    _prof_ai_count = _prof_jpeg_cnt = 0

    try:
        while _running:
            # Paused: finalize the current segment once. If anyone is watching
            # live, keep feeding the preview at live_fps (no record I/O).
            if not _recording_enabled:
                if rec is not None:
                    print("[dvr] Gravação pausada.")
                    rec.close()
                    rec = None
                    with _stats_lock:
                        _stats["status"]          = "paused"
                        _stats["current_segment"] = ""
                # Quando pausado com watcher, faz live JPEG.
                # Se engine ativa, aproveita pra rodar inferência no mesmo frame.
                engine_snapshot = _engine
                if _live_watchers == 0 and engine_snapshot is None:
                    time.sleep(0.2)
                    continue
                time.sleep(jpeg_period)
                frame = cam.read()
                try:
                    if _live_watchers > 0:
                        jpeg = image.frame_to_jpeg(
                            frame,
                            quality=args.live_jpeg_quality,
                            scale=args.live_scale)
                        with _jpeg_lock:
                            _latest_jpeg = jpeg
                        last_jpeg_time = time.time()
                    if engine_snapshot is not None:
                        # Pausado => sem seg_start; usa wall-clock.
                        engine_snapshot.infer(frame,
                                              int(time.time() * 1000),
                                              args.width, args.height)
                        _publish_detections()
                finally:
                    cam.release()
                continue

            # Resumed: rebuild recorder and reset pacing.
            if rec is None:
                print("[dvr] Gravação retomada.")
                rec = _make_recorder(args)
                t_start   = time.time()
                frame_idx = 0
                next_tick = t_start
                with _stats_lock:
                    _stats["status"] = "recording"

            # Pace the loop to target fps.  A small positive drift is fine;
            # a large negative drift (we're behind) just means no sleep.
            now  = time.time()
            wait = next_tick - now
            if wait > 0:
                time.sleep(wait)
            next_tick += period

            # JPEG preview is opportunistic: only encode when the loop has
            # slack (wait > 0 meant we arrived ahead of schedule). This keeps
            # recording as the priority — live view degrades under load, never
            # the other way around.
            have_slack = wait > 0

            engine_snapshot = _engine

            # === [PROFILE] timing breakdown ===========================
            _t0 = time.time()
            frame = cam.read()
            _t_read = time.time() - _t0

            _t_send = _t_ai = _t_jpeg = 0.0
            try:
                # Prioridade 1 — gravação (sempre).
                _t1 = time.time()
                try:
                    rec.send_frame(frame)
                except Exception as e:
                    # VENC retornou erro (tipicamente ERR_VENC_BUSY 0xC0078012).
                    # Estado interno do encoder está corrompido — chamar
                    # rec.close() aqui levaria o driver pra D-state. Marcamos
                    # broken pra finally pular o close.
                    print(f"[dvr] recorder erro fatal: {e}", flush=True)
                    _recorder_broken = True
                    raise
                _t_send = time.time() - _t1
                # Prioridade 2 — inferência IA. Roda em TODO frame (sem gate
                # de slack) pra garantir que o JSONL do segmento fica denso
                # mesmo durante live mode. Tradeoff: se infer() for pesado, o
                # FPS-alvo pode cair — aceito porque IA > live na hierarquia.
                if engine_snapshot is not None:
                    current = rec.current_segment()
                    if current:
                        engine_snapshot.open_segment(current, args.out_dir)
                        seg_start = rec.segment_start_ms()
                        t_ms = int(time.time() * 1000) - seg_start \
                               if seg_start else 0
                        _t2 = time.time()
                        engine_snapshot.infer(frame, t_ms,
                                              args.width, args.height)
                        _t_ai = time.time() - _t2
                        _publish_detections()
                # Prioridade 3 — JPEG live. Re-checa slack APÓS a IA: só
                # encoda se o loop continua adiantado em relação ao tick.
                # Assim o JPEG cede CPU pra IA quando há aperto.
                if (_live_watchers > 0
                        and (now - last_jpeg_time) >= jpeg_period
                        and (next_tick - time.time()) > 0):
                    _t3 = time.time()
                    jpeg = image.frame_to_jpeg(
                        frame,
                        quality=args.live_jpeg_quality,
                        scale=args.live_scale)
                    _t_jpeg = time.time() - _t3
                    with _jpeg_lock:
                        _latest_jpeg = jpeg
                    last_jpeg_time = now
            finally:
                cam.release()
            # Acumula pra report periódico
            _prof_n    += 1
            _prof_read += _t_read
            _prof_send += _t_send
            _prof_ai   += _t_ai
            _prof_jpeg += _t_jpeg
            if _t_ai > 0:   _prof_ai_count += 1
            if _t_jpeg > 0: _prof_jpeg_cnt += 1

            frame_idx += 1
            now = time.time()
            elapsed = now - t_start
            fps_avg = frame_idx / elapsed if elapsed > 0 else 0.0

            current_seg = rec.current_segment()
            with _stats_lock:
                _stats["frame_count"]     = frame_idx
                _stats["fps"]             = round(fps_avg, 1)
                _stats["current_segment"] = current_seg

            # Segmento rotacionou (um anterior fechou OU o primeiro abriu):
            # reescreve manifest.js em background. No-op em cargas curtas.
            if current_seg and current_seg != last_segment:
                if last_segment:
                    _schedule_manifest(args.out_dir)
                last_segment = current_seg

            if now - t_report >= 5.0:
                print(f"  frame {frame_idx:6d}  fps={fps_avg:4.1f}  "
                      f"segment={current_seg or '(waiting keyframe)'}")
                # === [PROFILE] médias do intervalo (em ms por frame) =====
                if _prof_n > 0:
                    n = _prof_n
                    avg_read = 1000.0 * _prof_read / n
                    avg_send = 1000.0 * _prof_send / n
                    avg_ai   = (1000.0 * _prof_ai / _prof_ai_count) \
                                if _prof_ai_count else 0.0
                    avg_jpeg = (1000.0 * _prof_jpeg / _prof_jpeg_cnt) \
                                if _prof_jpeg_cnt else 0.0
                    print(f"  [prof] frames={n}  "
                          f"read={avg_read:5.1f}ms  "
                          f"send={avg_send:5.1f}ms  "
                          f"ai={avg_ai:5.1f}ms x{_prof_ai_count}  "
                          f"jpeg={avg_jpeg:5.1f}ms x{_prof_jpeg_cnt}  "
                          f"watchers={_live_watchers}",
                          flush=True)
                _prof_n = _prof_read = _prof_send = _prof_ai = _prof_jpeg = 0
                _prof_ai_count = _prof_jpeg_cnt = 0
                t_report = now

    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        with _stats_lock:
            _stats["status"] = f"error: {exc}"
    finally:
        _running = False
        with _stats_lock:
            _stats["status"] = "stopped"
        # Acorda SSE subscribers cedo pra não segurar threads.
        with _sse_cond:
            _sse_cond.notify_all()

        # _recorder_broken: send_frame retornou erro fatal (tipicamente
        # CVI_ERR_VENC_BUSY 0xC0078012). Chamar rec.close() nesse caso
        # levaria o driver pra D-state (uninterruptible) — só reboot
        # recupera. Pulamos o close e deixamos o kernel limpar o FD no
        # exit do processo (que pode ou não conseguir limpar o VENC).
        if rec is not None:
            if _recorder_broken:
                print("[dvr] recorder em estado ruim — pulando close pra "
                      "evitar D-state.", flush=True)
            else:
                print("[dvr] fechando recorder...", flush=True)
                try:
                    rec.close()
                except Exception as e:
                    print(f"[dvr] rec.close exception: {e}", flush=True)
                print("[dvr] recorder fechado.", flush=True)
        print("[dvr] fechando câmera...", flush=True)
        try:
            cam.close()
        except Exception as e:
            print(f"[dvr] cam.close exception: {e}", flush=True)
        print("[dvr] câmera fechada.", flush=True)
        with _engine_lock:
            if _engine is not None:
                print("[dvr] fechando engine...", flush=True)
                try:
                    _engine.close()
                except Exception as e:
                    print(f"[dvr] engine.close exception: {e}", flush=True)
                print("[dvr] engine fechada.", flush=True)

    print(f"\nFinalizado. {frame_idx} frames.")


# ─── File serving helpers ────────────────────────────────────────────────────

# basename must match the recorder's naming: dvr_YYYY-MM-DD_HH-MM-SS.mp4
_NAME_RE = re.compile(
    r"^dvr_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.mp4$")


def _scan_segments(out_dir):
    """Return list of dicts for all closed MP4 segments in out_dir.
    Exclude the segment currently being written — its moov atom is not yet
    on disk and clicking it in the player would hang/error."""
    try:
        names = os.listdir(out_dir)
    except OSError:
        return []
    with _stats_lock:
        current = _stats.get("current_segment", "") or ""
    current_base = os.path.basename(current) if current else ""
    out = []
    for n in names:
        m = _NAME_RE.match(n)
        if not m:
            continue
        if n == current_base:
            continue
        full = os.path.join(out_dir, n)
        try:
            st = os.stat(full)
        except OSError:
            continue
        thumb = n[:-4] + ".jpg"
        thumb_path = os.path.join(out_dir, thumb)
        out.append({
            "name":        n,
            "thumbnail":   thumb if os.path.exists(thumb_path) else "",
            "size":        st.st_size,
            "mtime":       int(st.st_mtime),
            "label":       "{}-{}-{} {}:{}:{}".format(*m.groups()),
        })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


_manifest_lock = threading.Lock()

def _write_manifest(out_dir):
    """Gera manifest.js no out_dir com a lista de segmentos fechados
    (apenas metadata: name/thumb/size/mtime/label). Detecções por segmento
    são carregadas on-demand pelo viewer via /files/<stem>.jsonl quando o
    usuário clica num segmento, evitando inflar o manifest.
    Executa sob lock pra evitar entrelaçar rebuilds concorrentes."""
    if not _manifest_lock.acquire(blocking=False):
        # Já tem outra thread reescrevendo — skip, ela cobre o estado atual.
        return
    try:
        segs = _scan_segments(out_dir)
        out = [{
            "name":  s["name"],
            "thumb": s["thumbnail"] or "",
            "size":  s["size"],
            "mtime": s["mtime"],
            "label": s["label"],
        } for s in segs]
        payload = {
            "folder":       os.path.basename(out_dir.rstrip("/")) or "dvr",
            "generated_at": int(time.time() * 1000),
            "segments":     out,
        }
        # Escrita direta: ntfs-3g (FUSE) tem limitações pra rename atômico,
        # então evitamos o tmp+rename. _manifest_lock garante serialização.
        final = os.path.join(out_dir, "manifest.js")
        with open(final, "w") as f:
            f.write("window.DVR_MANIFEST = ")
            json.dump(payload, f, separators=(",", ":"))
            f.write(";\n")
    except Exception as e:
        print(f"[dvr] erro escrevendo manifest: {e}", flush=True)
    finally:
        _manifest_lock.release()


def _schedule_manifest(out_dir):
    """Dispara _write_manifest em thread daemon pra não bloquear o loop."""
    threading.Thread(target=_write_manifest, args=(out_dir,),
                     daemon=True).start()


def _clear_segments(out_dir):
    """Delete every .mp4/.jpg/.jsonl that matches the dvr naming pattern.
    Skips the segment currently being written (held open by the recorder)."""
    with _stats_lock:
        current = _stats.get("current_segment", "") or ""
    current_base = os.path.basename(current) if current else ""
    current_stem = current_base[:-4] if current_base.endswith(".mp4") else ""

    removed = 0
    errors  = 0
    try:
        names = os.listdir(out_dir)
    except OSError:
        return {"removed": 0, "errors": 0, "skipped_current": current_base}

    for n in names:
        if n.endswith(".mp4"):
            if not _NAME_RE.match(n):
                continue
            stem = n[:-4]
        elif n.endswith(".jpg") or n.endswith(".jsonl"):
            stem = n.rsplit(".", 1)[0]
            if not _NAME_RE.match(stem + ".mp4"):
                continue
        else:
            continue
        if current_stem and stem == current_stem:
            continue
        try:
            os.remove(os.path.join(out_dir, n))
            removed += 1
        except OSError:
            errors += 1
    return {"removed": removed, "errors": errors,
            "skipped_current": current_base}


def _send_file(handler, path, content_type):
    """Serve a static file with HTTP Range support (required for <video>)."""
    try:
        st = os.stat(path)
    except OSError:
        handler.send_response(404)
        handler.end_headers()
        return

    size  = st.st_size
    rng   = handler.headers.get("Range")
    start = 0
    end   = size - 1
    status = 200

    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m:
            s, e = m.groups()
            if s:
                start = int(s)
            if e:
                end = int(e)
            if start >= size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.end_headers()
                return
            if end >= size:
                end = size - 1
            status = 206

    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type",   content_type)
    handler.send_header("Accept-Ranges",  "bytes")
    handler.send_header("Content-Length", str(length))
    if status == 206:
        handler.send_header("Content-Range",
                            f"bytes {start}-{end}/{size}")
    handler.end_headers()

    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            remaining -= len(chunk)


# ─── Web server ──────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TDL DVR</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0a0c0f;
    --panel: #111318;
    --border: #1e2330;
    --accent: #00e5ff;
    --rec: #ff3b30;
    --text: #c8d0e0;
    --dim: #4a5268;
    --mono: 'Menlo','Consolas',monospace;
  }
  body {
    background: var(--bg); color: var(--text);
    font-family: system-ui, sans-serif; font-size: 14px;
    min-height: 100vh; display: flex; flex-direction: column;
    overflow: hidden;
  }
  header {
    display: flex; align-items: center; gap: 16px;
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel); flex-shrink: 0;
  }
  h1 {
    font-size: 13px; font-weight: 600; letter-spacing: .2em;
    text-transform: uppercase; color: var(--accent);
  }
  .rec-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--rec); box-shadow: 0 0 10px var(--rec);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  .rec-dot.idle { background: var(--dim); box-shadow: none; animation: none; }
  .hdr-info {
    font-family: var(--mono); font-size: 11px; color: var(--dim);
    margin-left: auto; display: flex; gap: 18px;
  }
  .hdr-info span.val { color: var(--text); }
  .hdr-btn {
    background: transparent; color: var(--accent);
    border: 1px solid var(--accent); border-radius: 3px;
    padding: 4px 12px; font: inherit; font-size: 10px; font-weight: 600;
    letter-spacing: .2em; text-transform: uppercase;
    cursor: pointer; transition: all .15s;
  }
  .hdr-btn:hover { background: var(--accent); color: #0b0d12; }
  .hdr-btn.active {
    background: var(--rec); border-color: var(--rec); color: #fff;
    box-shadow: 0 0 8px var(--rec);
  }
  .hdr-btn.paused {
    color: #ffc857; border-color: #ffc857;
  }
  .hdr-btn.paused:hover { background: #ffc857; color: #0b0d12; }
  .hdr-sel {
    background: var(--panel); color: #ffffff;
    border: 1px solid var(--border); border-radius: 3px;
    padding: 4px 8px; font: inherit; font-size: 11px;
    max-width: 220px; cursor: pointer;
  }
  .hdr-sel:focus { outline: none; border-color: var(--accent); }
  /* Options no dropdown: força estilo dark — browsers senão caem no
     tema nativo do SO (frequentemente branco) e texto fica ilegível. */
  .hdr-sel option {
    background: var(--panel);
    color: #ffffff;
  }
  #overlay {
    position: absolute; pointer-events: none;
    top: 0; left: 0; width: 100%; height: 100%;
    display: none;
  }
  #overlay.on { display: block; }
  .player-wrap { position: relative; }
  .player-col {
    display: flex; flex-direction: column;
    min-width: 0; min-height: 0; overflow: hidden;
  }

  main {
    display: grid; grid-template-columns: 1fr;
    flex: 1 1 0; min-height: 0; overflow: hidden;
  }
  .player-wrap {
    background: #000; display: flex; align-items: center;
    justify-content: center; padding: 20px; overflow: hidden;
    flex: 1 1 0; min-height: 0;
  }
  .player-wrap video {
    max-width: 100%; max-height: 100%;
    background: #000;
  }
  /* Double-buffer: dois vídeos sobrepostos. Visibilidade controlada via
     inline style (JS) pra não conflitar com lógica de live-mode. */
  .vid {
    position: absolute; inset: 0; margin: auto;
    max-width: 100%; max-height: 100%;
    object-fit: contain;
  }

  /* Controles abaixo do player, sempre visíveis, sem overlap. */
  .ctrls {
    background: var(--panel); border-top: 1px solid var(--border);
    padding: 10px 14px; color: var(--text);
    font-family: var(--mono); font-size: 12px;
    flex-shrink: 0;
  }
  .ctrls-seek { margin-bottom: 6px; }
  #seek {
    width: 100%; height: 14px;
    -webkit-appearance: none; appearance: none;
    background: transparent; cursor: pointer;
  }
  #seek::-webkit-slider-runnable-track {
    height: 4px; background: rgba(255,255,255,.25); border-radius: 2px;
  }
  #seek::-moz-range-track {
    height: 4px; background: rgba(255,255,255,.25); border-radius: 2px;
  }
  #seek::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); margin-top: -5px;
    box-shadow: 0 0 4px rgba(0,229,255,.6);
  }
  #seek::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%; border: none; background: var(--accent);
  }
  .ctrls-row { display: flex; align-items: center; gap: 4px; }
  .cbtn {
    background: transparent; border: none; color: #fff;
    font-size: 18px; line-height: 1; cursor: pointer;
    padding: 6px 10px; border-radius: 4px; transition: background .15s;
  }
  .cbtn:hover { background: rgba(255,255,255,.15); }
  .cbtn-big { font-size: 22px; }
  .ctime { margin-left: 8px; color: #d4d8df; white-space: nowrap; }
  .cgrow { flex: 1; }
  .cvol {
    width: 80px; height: 14px; margin: 0 6px;
    -webkit-appearance: none; appearance: none;
    background: transparent; cursor: pointer;
  }
  .cvol::-webkit-slider-runnable-track {
    height: 3px; background: rgba(255,255,255,.3); border-radius: 2px;
  }
  .cvol::-moz-range-track {
    height: 3px; background: rgba(255,255,255,.3); border-radius: 2px;
  }
  .cvol::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 10px; height: 10px; border-radius: 50%;
    background: #fff; margin-top: -4px;
  }
  .cvol::-moz-range-thumb {
    width: 10px; height: 10px; border-radius: 50%; border: none; background: #fff;
  }
  .no-sel {
    color: var(--dim); font-family: var(--mono); font-size: 13px;
    text-align: center;
  }

  #clear-all:hover:not(:disabled) { color: #ff6b6b; border-color: #ff6b6b; }
  #clear-all:disabled { opacity: .4; cursor: not-allowed; }

  /* Barra de thumbnails (acesso rápido) na base */
  .quickbar {
    border-top: 1px solid var(--border); background: var(--panel);
    padding: 8px 10px; flex-shrink: 0;
    display: flex; gap: 6px; overflow-x: auto; scrollbar-width: thin;
  }
  .quickbar::-webkit-scrollbar { height: 6px; }
  .quickbar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .qb-item {
    flex-shrink: 0; cursor: pointer; position: relative;
    width: 120px; height: 68px; border: 2px solid transparent;
    border-radius: 4px; overflow: hidden; background: #000;
  }
  .qb-item.active { border-color: var(--accent); }
  .qb-item img { width: 100%; height: 100%; object-fit: cover; }
  .qb-item .label {
    position: absolute; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.65); color: var(--text);
    font-family: var(--mono); font-size: 9px; padding: 2px 4px;
    text-align: center;
  }
  .qb-item.placeholder {
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 10px; color: var(--dim);
  }

  .quickbar-empty {
    padding: 22px 16px; color: var(--dim); font-size: 12px;
    text-align: center; font-family: var(--mono);
  }
</style>
</head>
<body>
<header>
  <div class="rec-dot idle" id="rec-dot"></div>
  <h1>TDL DVR</h1>
  <div class="hdr-info">
    <span>STATUS: <span class="val" id="hdr-status">--</span></span>
    <span>FPS: <span class="val" id="hdr-fps">--</span></span>
    <span>FRAMES: <span class="val" id="hdr-frames">0</span></span>
    <span>SEGMENTS: <span class="val" id="hdr-segs">0</span></span>
    <span>AI: <span class="val" id="hdr-ai">0</span></span>
  </div>
  <select id="model-sel" class="hdr-sel" title="Modelo de IA">
    <option value="">— sem IA —</option>
  </select>
  <button id="auto-toggle" type="button" class="hdr-btn active"
          title="Reproduz o próximo segmento ao terminar">AUTO</button>
  <button id="ai-toggle"   type="button" class="hdr-btn" title="Mostrar overlay de IA">IA</button>
  <button id="rec-btn"     type="button" class="hdr-btn">PAUSAR</button>
  <button id="live-btn"    type="button" class="hdr-btn">AO VIVO</button>
  <button id="clear-all"   type="button" class="hdr-btn"
          title="Apagar todas as gravações do SD">LIMPAR</button>
</header>

<main>
  <div class="player-col">
    <div class="player-wrap">
      <div class="no-sel" id="no-sel">
        Selecione um segmento na barra abaixo.
      </div>
      <video id="player-a" class="vid" playsinline preload="auto" style="visibility:hidden"></video>
      <video id="player-b" class="vid" playsinline preload="auto" style="visibility:hidden"></video>
      <img id="live-img" alt="Live preview" style="display:none;max-width:100%;max-height:100%;object-fit:contain"/>
      <canvas id="overlay"></canvas>
    </div>
    <div class="ctrls" id="ctrls">
      <div class="ctrls-seek">
        <input type="range" id="seek" min="0" max="100" step="0.01" value="0">
      </div>
      <div class="ctrls-row">
        <button class="cbtn" id="btn-prev" title="Anterior">&#x23EE;</button>
        <button class="cbtn cbtn-big" id="btn-play" title="Play/Pause">&#x25B6;</button>
        <button class="cbtn" id="btn-next" title="Próximo">&#x23ED;</button>
        <span class="ctime" id="ctime">0:00 / 0:00</span>
        <span class="cgrow"></span>
        <button class="cbtn" id="btn-mute" title="Mudo">&#x1F509;</button>
        <input type="range" id="cvol" class="cvol" min="0" max="1" step="0.01" value="1">
        <button class="cbtn" id="btn-fs" title="Tela cheia">&#x26F6;</button>
      </div>
    </div>
  </div>

</main>

<div class="quickbar" id="quickbar"></div>

<script>
let _selected = "";
let _liveOn   = false;
let _autoOn   = true;   // autoplay de continuidade
let _items    = [];     // última lista do /api/list (ordenada newest→oldest)
// Double-buffer: `player` aponta pro visível, `playerNext` pré-carrega
// o próximo segmento. Na hora do swap, flipam as classes `.active` e
// as variáveis apontam pro oposto — zero flicker.
let player     = document.getElementById('player-a');
let playerNext = document.getElementById('player-b');
const noSel    = document.getElementById('no-sel');
const quickbar = document.getElementById('quickbar');
const liveImg  = document.getElementById('live-img');
const liveBtn  = document.getElementById('live-btn');

function stopLive() {
  if (!_liveOn) return;
  _liveOn = false;
  liveImg.removeAttribute('src');  // closes the MJPEG connection
  liveImg.style.display = 'none';
  liveBtn.classList.remove('active');
  liveBtn.textContent = 'AO VIVO';
  if (_selected) {
    // Retoma o player que foi pausado/ocultado em startLive.
    player.style.visibility = 'visible';
    player.play().catch(() => { /* autoplay may be blocked */ });
  } else {
    noSel.style.display = 'flex';
  }
  if (typeof syncCtrlsVisibility === 'function') syncCtrlsVisibility();
}

function startLive() {
  if (_liveOn) return;
  _liveOn = true;
  if (_selected) {
    player.pause();
    player.style.visibility = 'hidden';
  }
  noSel.style.display = 'none';
  // Cache-bust so a reconnect actually re-hits the endpoint.
  liveImg.src = '/api/live.mjpg?t=' + Date.now();
  liveImg.style.display = 'block';
  liveBtn.classList.add('active');
  liveBtn.textContent = 'PARAR';
  if (typeof syncCtrlsVisibility === 'function') syncCtrlsVisibility();
}

liveBtn.addEventListener('click', () => _liveOn ? stopLive() : startLive());

const recBtn = document.getElementById('rec-btn');
async function toggleRecording() {
  recBtn.disabled = true;
  const url = recBtn.textContent === 'PAUSAR'
    ? '/api/record/pause' : '/api/record/resume';
  try {
    const r = await fetch(url, { method: 'POST' });
    const j = await r.json();
    updateRecBtn(j.recording);
    await poll();
  } catch (_) {}
  recBtn.disabled = false;
}
function updateRecBtn(recording) {
  recBtn.textContent = recording ? 'PAUSAR' : 'RETOMAR';
  recBtn.classList.toggle('paused', !recording);
}
recBtn.addEventListener('click', toggleRecording);

// ────────── Autoplay de continuidade ──────────
// Ao terminar um segmento, seleciona o próximo cronológico.
const autoToggle = document.getElementById('auto-toggle');
autoToggle.addEventListener('click', () => {
  _autoOn = !_autoOn;
  autoToggle.classList.toggle('active', _autoOn);
});
// Listeners anexados aos DOIS <video> (double-buffer). Só age quando
// vem do ativo — eventos do que está pré-carregando são ignorados.
function attachPlayerListeners(el) {
  el.addEventListener('ended', (e) => {
    if (e.target !== player) return;
    if (!_autoOn || !_selected || _liveOn) return;
    const idx = _items.findIndex(it => it.name === _selected);
    if (idx <= 0) return;
    const next = _items[idx - 1];
    setTimeout(() => {
      const elItem = quickbar.querySelector(`.qb-item[data-name="${next.name}"]`);
      if (elItem && elItem.scrollIntoView)
        elItem.scrollIntoView({block: 'nearest', inline: 'center'});
    }, 50);
    selectSegment(next.name);
  });
  el.addEventListener('loadedmetadata', (e) => {
    if (e.target !== player) return;
    resizeOverlay();
  });
  el.addEventListener('timeupdate', (e) => {
    if (e.target !== player) return;
    if (_selected && _aiOn) renderFromVideo();
  });
}
attachPlayerListeners(document.getElementById('player-a'));
attachPlayerListeners(document.getElementById('player-b'));

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/(1024*1024)).toFixed(1) + ' MB';
}

function srcUrl(name) { return '/files/' + encodeURIComponent(name); }

function highlight(name) {
  for (const el of document.querySelectorAll('.qb-item')) {
    el.classList.toggle('active', el.dataset.name === name);
  }
}

// Pré-carrega no player oculto o segmento cronologicamente posterior a
// `currentName` (lista é newest-first → "próximo" = índice - 1).
function preloadNext(currentName) {
  playerNext.pause();
  const idx = _items.findIndex(it => it.name === currentName);
  if (idx <= 0) { playerNext.removeAttribute('src'); return; }
  const next = _items[idx - 1];
  playerNext.src = srcUrl(next.name);
  playerNext.load();
}

// Swap instantâneo: playerNext (já pré-carregado) vira ativo.
function swapToNext(name) {
  player.style.visibility     = 'hidden';
  playerNext.style.visibility = 'visible';
  const old = player;
  player = playerNext;
  playerNext = old;
  playerNext.pause();
  playerNext.removeAttribute('src');
  player.play().catch(() => {});
  _selected = name;
  highlight(name);
  resizeOverlay();
  if (_aiOn) renderFromVideo();
  preloadNext(name);
}

function selectSegment(name) {
  if (_liveOn) stopLive();
  // Se o player oculto já tem esse vídeo pré-carregado, swap instantâneo.
  const want = srcUrl(name);
  if (playerNext.src && playerNext.src.endsWith(want)) {
    swapToNext(name);
    player.style.visibility = 'visible';
    noSel.style.display     = 'none';
    return;
  }
  _selected = name;
  player.src = want;
  player.style.visibility = 'visible';
  noSel.style.display     = 'none';
  player.play().catch(() => { /* autoplay may be blocked */ });
  highlight(name);
  preloadNext(name);
}

function renderList(items) {
  _items = items;
  document.getElementById('hdr-segs').textContent = items.length;

  if (items.length === 0) {
    quickbar.innerHTML =
      '<div class="quickbar-empty">Nenhum segmento ainda.</div>';
    return;
  }

  // loading="lazy" + decoding="async" → o browser só baixa o thumb quando
  // ele entra no viewport horizontal da quickbar (rolagem lateral).
  quickbar.innerHTML = items.map(it => {
    const time = it.label.split(' ')[1] || it.label;
    const tip  = `${it.label} — ${fmtBytes(it.size)}`;
    return `
    <div class="qb-item${it.name === _selected ? ' active' : ''}"
         data-name="${it.name}" title="${tip}">
      ${it.thumbnail
        ? `<img loading="lazy" decoding="async"
                src="/files/${encodeURIComponent(it.thumbnail)}" alt="">`
        : ''}
      <div class="label">${time}</div>
    </div>`;
  }).join('');
  for (const el of quickbar.querySelectorAll('.qb-item')) {
    el.addEventListener('click', () => selectSegment(el.dataset.name));
  }
}

async function poll() {
  try {
    const [lr, sr] = await Promise.all([
      fetch('/api/list').then(r => r.json()),
      fetch('/api/status').then(r => r.json()),
    ]);
    renderList(lr.items || []);
    const st = sr || {};
    document.getElementById('hdr-status').textContent = st.status || '--';
    document.getElementById('hdr-fps').textContent    = st.fps ?? '--';
    document.getElementById('hdr-frames').textContent = st.frame_count ?? 0;
    document.getElementById('rec-dot').classList.toggle(
      'idle', st.status !== 'recording');
    if (!recBtn.disabled)
      updateRecBtn(st.status === 'recording');
  } catch (e) { /* ignore */ }
}

document.getElementById('clear-all').addEventListener('click', async () => {
  const btn = document.getElementById('clear-all');
  if (!confirm('Apagar todas as gravações do SD card?')) return;
  btn.disabled = true;
  const oldTxt = btn.textContent;
  btn.textContent = 'Apagando…';
  try {
    const r = await fetch('/api/clear-all', { method: 'POST' });
    const j = await r.json();
    if (_selected) {
      player.pause(); player.removeAttribute('src'); player.load();
      player.style.visibility = 'hidden'; noSel.style.display = 'flex';
      _selected = "";
    }
    await poll();
    let msg = `${j.removed} arquivo(s) removido(s).`;
    if (j.errors) msg += ` ${j.errors} erro(s).`;
    if (j.skipped_current) msg += ` Segmento atual preservado.`;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = oldTxt; btn.disabled = false; }, 2500);
  } catch (e) {
    btn.textContent = 'Erro';
    setTimeout(() => { btn.textContent = oldTxt; btn.disabled = false; }, 2000);
  }
});

// ─── AI overlay ────────────────────────────────────────────────────────────
const modelSel = document.getElementById('model-sel');
const aiToggle = document.getElementById('ai-toggle');
const overlay  = document.getElementById('overlay');
const octx     = overlay.getContext('2d');

let _aiOn         = false;
let _currentModel = "";
let _liveDets     = [];
let _segDets      = [];
let _sseStream    = null;

const CLS_COLORS = ["#22c55e","#3b82f6","#f59e0b","#ef4444","#a855f7",
                    "#06b6d4","#ec4899","#84cc16","#f97316","#14b8a6"];
function colorForClass(id) { return CLS_COLORS[Math.abs(id|0) % CLS_COLORS.length]; }

function resizeOverlay() {
  let host = null;
  if (player.style.visibility !== 'hidden' && player.videoWidth) host = player;
  else if (liveImg.style.display !== 'none') host = liveImg;
  if (!host) {
    console.log('resizeOverlay: no host');
    overlay.width = 0; overlay.height = 0; overlay._draw = null; return;
  }
  // Usa o rect visual do host (e do canvas) no documento pra alinhar
  // exatamente canvas-bitmap com área renderizada da imagem.  Isso evita
  // o problema de canvas CSS=100% do container ser maior que a imagem
  // (letterbox flex) — nesse caso o bitmap é esticado e bboxes saem fora.
  const hostRect = host.getBoundingClientRect();
  const wrapRect = overlay.parentElement.getBoundingClientRect();
  const W = Math.round(hostRect.width);
  const H = Math.round(hostRect.height);
  if (!W || !H) {
    console.log('resizeOverlay: zero dim', {W, H, host: host.id});
    overlay.width = 0; overlay.height = 0; overlay._draw = null; return;
  }
  // Posiciona o canvas EXATAMENTE sobre o host (em vez de 100%/100% do wrap).
  overlay.style.left  = (hostRect.left - wrapRect.left) + 'px';
  overlay.style.top   = (hostRect.top  - wrapRect.top)  + 'px';
  overlay.style.width = W + 'px';
  overlay.style.height= H + 'px';
  overlay.width  = W;
  overlay.height = H;
  // Como o canvas agora cobre só a imagem, dx/dy=0 e dw/dh=W/H.
  overlay._draw  = { dx: 0, dy: 0, dw: W, dh: H };
  console.log('resizeOverlay:', host.id, {W, H,
                                           left: overlay.style.left,
                                           top:  overlay.style.top});
}

// COCO-17 human pose skeleton (pares de índice 0-based).
const COCO17_SKELETON = [
  [0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,6],[5,7],[7,9],
  [6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
];
const KP_SCORE_THR = 0.3;

function drawDets(dets) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!overlay._draw || !_aiOn) return;
  const { dx, dy, dw, dh } = overlay._draw;
  octx.lineWidth = 2;
  octx.font = '12px ui-monospace, Menlo, monospace';
  octx.textBaseline = 'top';

  for (const d of dets) {
    const color = colorForClass(d.id || 0);
    const kps   = d.kps || [];
    // Lane/linha: 2 keypoints implicam polilínia — bbox/label seriam ruído,
    // então pulamos. (LSTR_DET_LANE cai aqui.)
    const isLine = kps.length === 2;

    // Bbox + label
    if (d.bb && !isLine) {
      const [x1,y1,x2,y2] = d.bb;
      const X = dx + x1*dw, Y = dy + y1*dh;
      const BW = (x2-x1)*dw, BH = (y2-y1)*dh;
      octx.strokeStyle = color;
      octx.strokeRect(X, Y, BW, BH);
      const label = `${d.cls} ${Math.round((d.s||0)*100)}%`;
      const tw = octx.measureText(label).width + 8;
      octx.fillStyle = color;
      octx.fillRect(X, Math.max(0, Y-16), tw, 16);
      octx.fillStyle = '#0b0d12';
      octx.fillText(label, X+4, Math.max(0, Y-15));
    }

    // Keypoints + skeleton/polyline
    if (kps.length) {
      octx.strokeStyle = color;
      octx.lineWidth   = isLine ? 3 : 2;

      if (kps.length === 17) {
        // COCO-17 human pose
        for (const [a, b] of COCO17_SKELETON) {
          const pa = kps[a], pb = kps[b];
          if (!pa || !pb) continue;
          if ((pa[2]||0) < KP_SCORE_THR || (pb[2]||0) < KP_SCORE_THR) continue;
          octx.beginPath();
          octx.moveTo(dx + pa[0]*dw, dy + pa[1]*dh);
          octx.lineTo(dx + pb[0]*dw, dy + pb[1]*dh);
          octx.stroke();
        }
      } else if (isLine || (!d.bb && kps.length <= 8)) {
        // Polilínia: lane (2 pts) ou detecções só-landmarks pequenas.
        octx.beginPath();
        let started = false;
        for (const p of kps) {
          if (!p) continue;
          if ((p[2]||0) < KP_SCORE_THR) continue;
          const X = dx + p[0]*dw, Y = dy + p[1]*dh;
          if (!started) { octx.moveTo(X, Y); started = true; }
          else           octx.lineTo(X, Y);
        }
        octx.stroke();
      }

      // Pontos (exceto para lanes — a linha sozinha é mais limpa)
      if (!isLine) {
        octx.fillStyle = color;
        for (const p of kps) {
          if (!p) continue;
          if ((p[2]||0) < KP_SCORE_THR) continue;
          octx.beginPath();
          octx.arc(dx + p[0]*dw, dy + p[1]*dh, 3, 0, Math.PI*2);
          octx.fill();
        }
      }
    }
  }
}

function findSegDets(tMs) {
  if (!_segDets.length) return [];
  // Acha a maior entrada com t <= tMs.
  let lo = 0, hi = _segDets.length - 1, prev = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (_segDets[mid].t <= tMs) { prev = mid; lo = mid+1; }
    else                         { hi = mid-1; }
  }
  // Compara com a próxima (se existir) e usa a mais próxima de tMs nas
  // duas direções. Reduz pela metade a latência percebida do overlay
  // (sem isso só se vê detecções "passadas", nunca as do frame seguinte).
  let nearest = prev;
  if (prev + 1 < _segDets.length) {
    const dPrev = tMs - _segDets[prev].t;
    const dNext = _segDets[prev + 1].t - tMs;
    if (dNext < dPrev) nearest = prev + 1;
  }
  const e = _segDets[nearest];
  return (e && Math.abs(tMs - e.t) < 500) ? (e.dets || []) : [];
}

function renderFromVideo() {
  if (!_aiOn) return;
  drawDets(findSegDets(player.currentTime * 1000));
}

function setOverlayVisible(on) {
  _aiOn = on;
  overlay.classList.toggle('on', on);
  aiToggle.classList.toggle('active', on);
  if (!on) octx.clearRect(0, 0, overlay.width, overlay.height);
  else { resizeOverlay();
         if (_liveOn) drawDets(_liveDets); else if (_selected) renderFromVideo(); }
}

aiToggle.addEventListener('click', () => setOverlayVisible(!_aiOn));

// Liga overlay por padrão. Tem que passar pelo setOverlayVisible() pra
// aplicar a classe .on no canvas (display:block) e chamar resizeOverlay().
setOverlayVisible(true);

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    const j = await r.json();
    const prev = _currentModel;
    _currentModel = j.current || "";
    modelSel.innerHTML = '<option value="">— sem IA —</option>';
    for (const m of (j.available || [])) {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = m.name + ' (' + (m.types || []).join(',') + ')';
      if (m.name === _currentModel) o.selected = true;
      modelSel.appendChild(o);
    }
    // Mostra o erro do último load falho no title do select.
    modelSel.title = j.error
      ? ('Falha ao carregar modelo: ' + j.error)
      : 'Modelo de IA';
    modelSel.style.borderColor = j.error ? '#ef4444' : '';
    if (j.error && j.error !== window._lastModelErr) {
      console.warn('Model load error:', j.error);
      window._lastModelErr = j.error;
    }
    // Se backend já tem modelo carregado (via --model) e overlay está off,
    // auto-ativa. Também reconecta SSE caso o estado tenha mudado.
    if (_currentModel && !_aiOn) setOverlayVisible(true);
    if (_currentModel !== prev) updateSseStream();
  } catch (_) {}
}

modelSel.addEventListener('change', async () => {
  const name = modelSel.value;
  modelSel.disabled = true;
  try {
    const r = await fetch('/api/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const j = await r.json();
    if (!j.ok) alert('Falha ao carregar modelo: ' + (j.error || '?'));
    _currentModel = j.name || "";
    if (!_currentModel) {
      _liveDets = []; _segDets = []; drawDets([]);
      document.getElementById('hdr-ai').textContent = '0';
    } else {
      // Auto-ativa overlay quando modelo é escolhido (pode desligar via botão IA).
      if (!_aiOn) setOverlayVisible(true);
    }
    updateSseStream();
  } finally {
    modelSel.disabled = false;
  }
});

function updateSseStream() {
  const want = _liveOn && !!_currentModel;
  if (want && !_sseStream) {
    _sseStream = new EventSource('/api/detections/live');
    _sseStream.onmessage = (e) => {
      try {
        const j = JSON.parse(e.data);
        _liveDets = j.dets || [];
        document.getElementById('hdr-ai').textContent = _liveDets.length;
        console.log('SSE dets:', _liveDets.length, j);
        if (_aiOn) {
          // Recalcula overlay geometry SEMPRE: o MJPEG só ganha dimensões
          // quando o primeiro frame renderiza, e o `load` event do <img>
          // pode disparar antes da SSE message chegar. Recalcular por
          // mensagem garante que a primeira detecção depois do MJPEG
          // ficar pronto vai pegar _draw válido. Custo: ~3 µs por msg.
          resizeOverlay();
          drawDets(_liveDets);
        }
      } catch (err) {
        console.error('SSE parse error:', err);
      }
    };
    _sseStream.onerror = () => { if (_sseStream){_sseStream.close();_sseStream=null;} };
  } else if (!want && _sseStream) {
    _sseStream.close(); _sseStream = null;
    _liveDets = [];
  }
}

async function loadSegJsonl(name) {
  _segDets = [];
  const stem = name.replace(/\.mp4$/, '');
  try {
    const r = await fetch('/files/' + encodeURIComponent(stem + '.jsonl'));
    if (!r.ok) {
      console.log('loadSegJsonl: HTTP', r.status, 'for', stem + '.jsonl');
      return;
    }
    const text = await r.text();
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      try {
        const o = JSON.parse(line);
        if (o.type === 'header') continue;
        if (typeof o.t === 'number') _segDets.push(o);
      } catch (_) {}
    }
    console.log('loadSegJsonl:', _segDets.length, 'entries for', stem);
  } catch (e) {
    console.error('loadSegJsonl error:', e);
  }
}

// Wrap selectSegment/startLive/stopLive p/ integrar overlay
const _origSelect    = selectSegment;
selectSegment = function(name) {
  _origSelect(name);
  loadSegJsonl(name).then(() => { if (_aiOn) renderFromVideo(); });
};
const _origStartLive = startLive;
startLive = function() { _origStartLive(); updateSseStream(); };
const _origStopLive  = stopLive;
stopLive = function() {
  _origStopLive();
  updateSseStream();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  // Player voltou — recalcula geometria do overlay no próximo frame
  // (depois do reflow do browser), senão fica com as dimensões menores
  // que o liveImg tinha.
  requestAnimationFrame(() => {
    resizeOverlay();
    if (_aiOn && _selected) renderFromVideo();
  });
};

// Redraw hooks
// rVFC usa `player` dinâmico → se rebind no ativo após cada swap.
// Anexo em ambos <video> garante que qualquer um que vire ativo já tem
// o loop em voo.
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  const tick = () => { if (_selected && _aiOn) renderFromVideo();
                       player.requestVideoFrameCallback(tick); };
  document.getElementById('player-a').requestVideoFrameCallback(tick);
  document.getElementById('player-b').requestVideoFrameCallback(tick);
}
// timeupdate fallback já está em attachPlayerListeners acima.
liveImg.addEventListener('load', () => { resizeOverlay();
                                         if (_aiOn && _liveOn) drawDets(_liveDets); });
window.addEventListener('resize', () => { resizeOverlay();
                                          if (_liveOn) drawDets(_liveDets);
                                          else if (_selected) renderFromVideo(); });

// ─── Custom controls ───────────────────────────────────────────────────────
const ctrls    = document.getElementById('ctrls');
const btnPrev  = document.getElementById('btn-prev');
const btnPlay  = document.getElementById('btn-play');
const btnNext  = document.getElementById('btn-next');
const btnMute  = document.getElementById('btn-mute');
const btnFs    = document.getElementById('btn-fs');
const seek     = document.getElementById('seek');
const cvol     = document.getElementById('cvol');
const ctimeEl  = document.getElementById('ctime');
const playerWrap = document.querySelector('.player-wrap');

function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${ss}`;
}
let _seekDragging = false;
function updateCtrls() {
  if (_liveOn) { ctrls.classList.remove('show'); return; }
  btnPlay.innerHTML = player.paused ? '&#x25B6;' : '&#x23F8;';
  if (isFinite(player.duration) && player.duration > 0) {
    seek.max = player.duration;
    if (!_seekDragging) seek.value = player.currentTime;
    ctimeEl.textContent = `${fmtTime(player.currentTime)} / ${fmtTime(player.duration)}`;
  } else {
    seek.max = 0; seek.value = 0;
    ctimeEl.textContent = fmtTime(player.currentTime);
  }
  const muted = player.muted || player.volume === 0;
  btnMute.innerHTML = muted ? '&#x1F507;' : '&#x1F509;';
  cvol.value = player.muted ? 0 : player.volume;
}

btnPlay.addEventListener('click', () => {
  if (player.paused) player.play().catch(() => {});
  else player.pause();
});
btnPrev.addEventListener('click', () => {
  if (!_selected) return;
  const idx = _items.findIndex(it => it.name === _selected);
  if (idx < 0 || idx >= _items.length - 1) return;
  selectSegment(_items[idx + 1].name);
});
btnNext.addEventListener('click', () => {
  if (!_selected) return;
  const idx = _items.findIndex(it => it.name === _selected);
  if (idx <= 0) return;
  selectSegment(_items[idx - 1].name);
});
btnMute.addEventListener('click', () => {
  const newMuted = !player.muted;
  player.muted = newMuted;
  playerNext.muted = newMuted;
  updateCtrls();
});
cvol.addEventListener('input', () => {
  const v = parseFloat(cvol.value);
  player.volume = v;       playerNext.volume = v;
  player.muted  = v === 0; playerNext.muted  = v === 0;
  updateCtrls();
});
seek.addEventListener('input', () => {
  _seekDragging = true;
  player.currentTime = parseFloat(seek.value);
});
seek.addEventListener('change', () => { _seekDragging = false; });
btnFs.addEventListener('click', () => {
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {
    (playerWrap.requestFullscreen || playerWrap.webkitRequestFullscreen)?.call(playerWrap);
  } else {
    (document.exitFullscreen || document.webkitExitFullscreen)?.call(document);
  }
});

for (const el of [document.getElementById('player-a'),
                  document.getElementById('player-b')]) {
  for (const ev of ['play','pause','timeupdate','volumechange','loadedmetadata','ratechange']) {
    el.addEventListener(ev, (e) => { if (e.target === player) updateCtrls(); });
  }
}

// Live mode esconde os controles de playback; no record mode sempre mostra.
function syncCtrlsVisibility() {
  ctrls.style.display = _liveOn ? 'none' : 'block';
}
syncCtrlsVisibility();

loadModels();
setInterval(loadModels, 30000);

setInterval(poll, 2000);
poll();
updateCtrls();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        _set_sched_idle()
        super().setup()

    def log_message(self, fmt, *args):
        pass  # quiet

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        """Server-Sent Events stream com as detecções correntes.
        Browsers consomem via `new EventSource('/api/detections/live')`."""
        self.send_response(200)
        self.send_header("Content-Type",   "text/event-stream")
        self.send_header("Cache-Control",  "no-cache")
        self.send_header("Connection",     "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_version = -1
        try:
            while _running:
                with _sse_cond:
                    if _sse_version == last_version:
                        # Aguarda novo dado (ou timeout, pra enviar heartbeat)
                        _sse_cond.wait(timeout=10.0)
                    if not _running:
                        break
                    version = _sse_version
                if version == last_version:
                    # Timeout — manda keepalive (SSE comentário)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                last_version = version
                eng = _engine
                payload = eng.latest() if eng else {"dets": [], "model": ""}
                data = ("data: " + json.dumps(payload) + "\n\n").encode()
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_mjpeg(self):
        """multipart/x-mixed-replace stream with the latest JPEG frame.
        Increments _live_watchers for the lifetime of the connection."""
        global _live_watchers
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace;boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "close")
        self.end_headers()
        _live_watchers += 1
        try:
            while _running:
                with _jpeg_lock:
                    jpeg = _latest_jpeg
                if not jpeg:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                 + jpeg + b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)   # bound at ~20 fps send cap
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _live_watchers -= 1

    def do_GET(self):
        path_only = self.path.split("?", 1)[0]
        if path_only == "/api/live.mjpg":
            self._serve_mjpeg()
            return

        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/list":
            self._send_json({"items": _scan_segments(_out_dir)})
            return

        if self.path == "/api/status":
            with _stats_lock:
                self._send_json(dict(_stats))
            return

        if self.path.startswith("/files/"):
            name = self.path[len("/files/"):]
            # sanitize: only allow the exact naming pattern we generate
            if "/" in name or ".." in name:
                self.send_response(400)
                self.end_headers()
                return
            if name.endswith(".mp4"):
                ctype = "video/mp4"
            elif name.endswith(".jpg"):
                ctype = "image/jpeg"
            elif name.endswith(".jsonl"):
                ctype = "application/jsonl"
            else:
                self.send_response(403)
                self.end_headers()
                return
            _send_file(self, os.path.join(_out_dir, name), ctype)
            return

        if self.path == "/api/models":
            if _args is None:
                self._send_json({"current": "", "available": [],
                                 "error": _engine_error})
                return
            models = video_inference.list_available_models(_args.model_dir)
            self._send_json({"current":   _engine_name,
                             "available": models,
                             "error":     _engine_error})
            return

        if self.path == "/api/model":
            cur = _engine.latest() if _engine else {"model": "", "dets": []}
            self._send_json(cur)
            return

        if path_only == "/api/detections/live":
            self._serve_sse()
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global _recording_enabled
        if self.path == "/api/clear-all":
            result = _clear_segments(_out_dir)
            _schedule_manifest(_out_dir)
            self._send_json(result)
            return
        if self.path == "/api/record/pause":
            _recording_enabled = False
            self._send_json({"recording": False})
            return
        if self.path == "/api/record/resume":
            _recording_enabled = True
            self._send_json({"recording": True})
            return
        if self.path == "/api/model":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                body = {}
            model_name = str(body.get("name", "") or "")
            model_type = str(body.get("type", "") or "")
            result = _set_engine(model_name, model_type, _args)
            self._send_json(result)
            return
        self.send_response(404)
        self.end_headers()


def _set_sched_idle():
    """Best-effort: lower the current thread's priority so the recording
    thread gets preference on the single-core CV181X."""
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError, PermissionError):
        try:
            os.nice(19)
        except (AttributeError, OSError):
            pass


def start_web(port):
    _set_sched_idle()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"Web UI disponível em http://0.0.0.0:{port}")
    srv.serve_forever()


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TDL DVR — grava câmera local em MP4")
    p.add_argument("--out-dir", default="/mnt/sd/dvr",
                   help="Diretório de saída no SD card (padrão: /mnt/sd/dvr)")
    p.add_argument("--segment-seconds", type=int, default=30,
                   dest="segment_seconds",
                   help="Duração de cada segmento em segundos (padrão: 30)")
    p.add_argument("--width",   type=int, default=1280,
                   help="Largura em pixels (padrão: 1280)")
    p.add_argument("--height",  type=int, default=720,
                   help="Altura em pixels (padrão: 720)")
    p.add_argument("--fps",     type=int, default=15,
                   help="Frame rate (padrão: 15). Deve corresponder ao "
                        "rate real do envio de frames; valores errados causam "
                        "MP4 com duração divergente do tempo real.")
    p.add_argument("--codec",   default="h264", choices=["h264", "h265"],
                   help="Codec de vídeo (padrão: h264)")
    p.add_argument("--bitrate", type=int, default=3072,
                   help="Bitrate em kbps (padrão: 3072)")
    p.add_argument("--gop",     type=int, default=15,
                   help="Keyframe interval em frames (padrão: 15)")
    p.add_argument("--mirror",  action="store_true",
                   help="Espelhar horizontalmente a imagem da câmera")
    p.add_argument("--flip",    action="store_true",
                   help="Inverter verticalmente a imagem da câmera")
    p.add_argument("--model", default="", dest="model_name",
                   help="NOME do modelo (chave do ModelType enum), ex: "
                        "YOLOV8_DET_COCO80, LSTR_DET_LANE, "
                        "KEYPOINT_YOLOV8POSE_PERSON17. NÃO é o caminho do "
                        ".cvimodel — é o identificador. Vazio = sem "
                        "inferência. Também pode ser trocado via web UI.")
    p.add_argument("--model-type", default="", dest="model_type",
                   help="Override do enum quando o nome no factory diverge "
                        "do enum. Normalmente deixar vazio.")
    p.add_argument("--model-dir", default="/root", dest="model_dir",
                   help="Diretório PAI dos .cvimodel. O factory resolve "
                        "<model_dir>/<plataforma>/<arquivo>. Padrão: /root "
                        "(→ /root/cv181x/<arquivo>_cv181x.cvimodel).")
    p.add_argument("--model-threshold", type=float, default=0.5,
                   dest="model_threshold",
                   help="Score threshold do modelo (padrão: 0.5)")
    p.add_argument("--live-fps", type=int, default=10, dest="live_fps",
                   help="FPS máximo do preview MJPEG (padrão: 10).")
    p.add_argument("--live-jpeg-quality", type=int, default=35,
                   dest="live_jpeg_quality",
                   help="Qualidade JPEG do preview, 1-100 (padrão: 35). "
                        "Valores baixos reduzem bytes/CPU, viabilizando fps "
                        "maior; o preview é só pra ajustes, qualidade é "
                        "secundária.")
    # scale=1.0 → HW JPEG encoder do CVI (quase zero CPU). scale<1 cai no
    # caminho SW (cv::cvtColor + resize + imencode), que rouba ~30-40ms da
    # CPU única do CV181X por frame e atrasa o pacing da gravação.
    p.add_argument("--live-scale", type=float, default=1.0,
                   dest="live_scale",
                   help="Fator de escala do preview (padrão: 1.0 = HW JPEG; "
                        "valores <1 forçam encode em SW e impactam a gravação)")
    p.add_argument("--web-port", type=int, default=9001, dest="web_port",
                   help="Porta do servidor web (padrão: 9001)")
    return p.parse_args()


def _is_mounted(path):
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == path:
                    return True
    except OSError:
        pass
    return False


def _wait_for_sd_mount(path="/mnt/sd", timeout_s=10):
    if _is_mounted(path):
        return True
    print(f"[dvr] {path} not mounted, waiting up to {timeout_s}s...", flush=True)
    for _ in range(timeout_s):
        time.sleep(1)
        if _is_mounted(path):
            print(f"[dvr] {path} mounted.", flush=True)
            return True
    print(f"[dvr] ERROR: {path} not mounted after {timeout_s}s. Aborting.",
          file=sys.stderr, flush=True)
    return False


def main():
    global _out_dir, _args
    args = parse_args()
    _args = args
    _out_dir = args.out_dir

    if not _wait_for_sd_mount("/mnt/sd", timeout_s=10):
        sys.exit(1)

    # Ensure the output dir exists up-front so the web listing doesn't 404.
    os.makedirs(_out_dir, exist_ok=True)

    # Carrega modelo inicial se passado via CLI. Falhas não abortam —
    # usuário pode carregar depois pela UI.
    if args.model_name:
        _set_engine(args.model_name, args.model_type, args)

    web = threading.Thread(target=start_web, args=(args.web_port,), daemon=True)
    web.start()

    record_loop(args)


if __name__ == "__main__":
    main()
