#!/usr/bin/env python3
"""
Sample: camera capture + detection + tracking + feature matching.

Tests all new tdl Python bindings:
  - tdl.image.Camera       (camera capture)
  - model.set_threshold / get_threshold
  - model.get_input_names / get_output_names
  - tdl.nn.Tracker         (multi-object tracking)
  - tdl.nn.Matcher         (feature matching / re-id)

Usage:
  python3 sample_camera_pipeline.py \\
      --det-model  /path/to/yolov8n_det_hand_face_person.cvimodel \\
      --feat-model /path/to/feature_cviface.cvimodel \\
      [--frames 30] [--width 1280] [--height 720] \\
      [--threshold 0.5] [--output-dir /tmp]

  feat-model is optional. If omitted, the matcher section is skipped.
"""

import sys
import os
import argparse
import numpy as np

import tdl
from tdl import image, nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_detections(detections):
    if not detections:
        print("  (no detections)")
        return
    for d in detections:
        print(f"  [{d['class_name']}]  score={d['score']:.3f}"
              f"  box=({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f})")


def print_tracks(tracks):
    if not tracks:
        print("  (no tracks)")
        return
    status_name = {0: "NEW", 1: "TRACKED", 2: "LOST", 3: "REMOVED"}
    for t in tracks:
        b = t["box_info"]
        print(f"  track_id={t['track_id']}  status={status_name.get(t['status'], '?')}"
              f"  box=({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f})"
              f"  vel=({t['velocity_x']:.1f},{t['velocity_y']:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model info & threshold control
# ─────────────────────────────────────────────────────────────────────────────

def test_model_info(model, threshold):
    separator("1. Model info & threshold control")

    print(f"  Default threshold : {model.get_threshold():.3f}")
    model.set_threshold(threshold)
    print(f"  After set_threshold({threshold}): {model.get_threshold():.3f}")

    input_names = model.get_input_names()
    output_names = model.get_output_names()
    print(f"  Input  tensors : {input_names}")
    print(f"  Output tensors : {output_names}")

    params = model.get_preprocess_parameters()
    print(f"  Preprocess mean : {params.get('mean')}")
    print(f"  Preprocess scale: {params.get('scale')}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Camera capture + detection
# ─────────────────────────────────────────────────────────────────────────────

def test_camera_detection(model, width, height, num_frames, output_dir):
    separator("2. Camera capture + detection")

    print(f"  Opening camera {width}x{height}, capturing {num_frames} frames ...")
    cam = image.Camera(width, height, image.ImageFormat.YUV420SP_VU)
    try:
        for frame_idx in range(num_frames):
            frame = cam.read()
            detections = model.inference(frame)

            w, h = frame.get_size()
            print(f"\n  Frame {frame_idx:03d}  ({w}x{h})  detections={len(detections)}")
            print_detections(detections)

            # Save first and last frame before releasing the buffer
            if frame_idx == 0 or frame_idx == num_frames - 1:
                out_path = os.path.join(output_dir, f"frame_{frame_idx:03d}.jpg")
                image.write(frame, out_path)
                print(f"  Saved: {out_path}")

            cam.release()
    finally:
        cam.close()
        print("\n  Camera closed.")

    return detections  # return last frame's detections for tracker demo


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Tracker (MOT-SORT)
# ─────────────────────────────────────────────────────────────────────────────

def test_tracker(model, width, height, num_frames):
    separator("3. Multi-object tracker (MOT-SORT)")

    # Configure tracker
    tracker = nn.Tracker(nn.TrackerType.MOT_SORT)
    tracker.set_img_size(width, height)

    cfg = tracker.get_track_config()
    print(f"  Default config:")
    print(f"    max_unmatched_times     = {cfg.max_unmatched_times}")
    print(f"    track_confirmed_frames  = {cfg.track_confirmed_frames}")
    print(f"    track_init_score_thresh = {cfg.track_init_score_thresh}")
    print(f"    high_score_thresh       = {cfg.high_score_thresh}")

    # Tweak and apply
    cfg.max_unmatched_times = 10
    cfg.track_confirmed_frames = 2
    tracker.set_track_config(cfg)
    print("  Applied custom config (max_unmatched=10, confirmed_frames=2)")

    print(f"\n  Opening camera {width}x{height}, tracking for {num_frames} frames ...")
    cam = image.Camera(width, height, image.ImageFormat.YUV420SP_VU)
    try:
        for frame_idx in range(num_frames):
            frame = cam.read()
            detections = model.inference(frame)
            tracks = tracker.track(detections, frame_idx)
            cam.release()
            active = [t for t in tracks if t["status"] in (0, 1)]  # NEW or TRACKED
            print(f"\n  Frame {frame_idx:03d}  dets={len(detections)}  active_tracks={len(active)}")
            print_tracks(active)
    finally:
        cam.close()
        print("\n  Camera closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Feature matcher (face re-id or image retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def test_matcher(feat_model, det_model, width, height, output_dir):
    separator("4. Feature matcher")

    matcher = nn.Matcher("cosine")
    print("  Matcher created (cosine similarity)")

    # Build gallery: capture a few frames, extract features for each detected face
    print(f"  Building gallery from camera ({width}x{height}) ...")
    gallery_features = []
    gallery_labels = []

    cam = image.Camera(width, height, image.ImageFormat.YUV420SP_VU)
    try:
        for frame_idx in range(5):
            frame = cam.read()
            detections = det_model.inference(frame)

            for det_idx, det in enumerate(detections[:2]):  # max 2 per frame
                x1 = int(max(0, det["x1"]))
                y1 = int(max(0, det["y1"]))
                w_box = int(det["x2"] - det["x1"])
                h_box = int(det["y2"] - det["y1"])
                if w_box < 16 or h_box < 16:
                    continue

                crop = image.crop(frame, (x1, y1, w_box, h_box))
                feat_result = feat_model.inference(crop)

                if isinstance(feat_result, np.ndarray):
                    gallery_features.append(feat_result)
                    label = f"frame{frame_idx}_det{det_idx}"
                    gallery_labels.append(label)
                    print(f"  Added gallery entry: {label}  "
                          f"feat_dim={feat_result.shape[0]}  dtype={feat_result.dtype}")

            cam.release()
    finally:
        cam.close()

    if not gallery_features:
        print("  No features extracted — skipping matcher test (no detections?)")
        return

    matcher.load_gallery(gallery_features)
    print(f"\n  Gallery loaded: {matcher.get_gallery_size()} entries, "
          f"dim={matcher.get_feature_dim()}")

    # Query: capture one more frame and search gallery
    print("\n  Querying with a new frame ...")
    cam = image.Camera(width, height, image.ImageFormat.YUV420SP_VU)
    try:
        frame = cam.read()
        detections = det_model.inference(frame)

        query_features = []
        for det in detections[:3]:
            x1 = int(max(0, det["x1"]))
            y1 = int(max(0, det["y1"]))
            w_box = int(det["x2"] - det["x1"])
            h_box = int(det["y2"] - det["y1"])
            if w_box < 16 or h_box < 16:
                continue
            crop = image.crop(frame, (x1, y1, w_box, h_box))
            feat_result = feat_model.inference(crop)
            if isinstance(feat_result, np.ndarray):
                query_features.append(feat_result)

        cam.release()
    finally:
        cam.close()

    if not query_features:
        print("  No query features extracted.")
        return

    topk = min(3, len(gallery_features))
    indices, scores = matcher.query(query_features, topk=topk)

    print(f"\n  Query results (top-{topk}):")
    for q_idx, (idx_list, score_list) in enumerate(zip(indices, scores)):
        print(f"  Query {q_idx}:")
        for rank, (idx, score) in enumerate(zip(idx_list, score_list)):
            label = gallery_labels[idx] if idx < len(gallery_labels) else f"idx_{idx}"
            print(f"    rank {rank+1}: gallery[{idx}]={label}  score={score:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Inference with runtime parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_with_params(model, width, height):
    separator("5. Inference with runtime parameters")

    cam = image.Camera(width, height, image.ImageFormat.YUV420SP_VU)
    try:
        frame = cam.read()
        results_default = model.inference(frame)
        results_strict  = model.inference(frame, {"score_threshold": 0.8})
        results_loose   = model.inference(frame, {"score_threshold": 0.2})
        cam.release()
    finally:
        cam.close()

    print(f"  Detections with default threshold : {len(results_default)}")
    print(f"  Detections with score_threshold=0.8: {len(results_strict)}")
    print(f"  Detections with score_threshold=0.2: {len(results_loose)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TDL Python bindings test pipeline")
    p.add_argument("--det-model",   required=True, help="Detection model .cvimodel path")
    p.add_argument("--feat-model",  default=None,  help="Feature extraction model .cvimodel path (optional)")
    p.add_argument("--frames",      type=int,   default=30,   help="Frames to capture per test (default: 30)")
    p.add_argument("--width",       type=int,   default=1280, help="Camera width  (default: 1280)")
    p.add_argument("--height",      type=int,   default=720,  help="Camera height (default: 720)")
    p.add_argument("--threshold",   type=float, default=0.5,  help="Detection threshold (default: 0.5)")
    p.add_argument("--output-dir",  default="/tmp", help="Directory for saved frames (default: /tmp)")
    p.add_argument("--model-type",  default="YOLOV8N_DET_HAND_FACE_PERSON",
                   help="ModelType enum name (default: YOLOV8N_DET_HAND_FACE_PERSON)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load detection model
    model_type = getattr(nn.ModelType, args.model_type, None)
    if model_type is None:
        print(f"Unknown ModelType: {args.model_type}")
        sys.exit(1)

    print(f"Loading detection model: {args.det_model}")
    det_model = nn.get_model(model_type, args.det_model)

    # Load feature model (optional)
    feat_model = None
    if args.feat_model:
        print(f"Loading feature model: {args.feat_model}")
        feat_model = nn.get_model(nn.ModelType.FEATURE_CVIFACE, args.feat_model)

    # Run tests
    test_model_info(det_model, args.threshold)

    test_camera_detection(det_model, args.width, args.height,
                          args.frames, args.output_dir)

    test_tracker(det_model, args.width, args.height, args.frames)

    if feat_model:
        test_matcher(feat_model, det_model, args.width, args.height, args.output_dir)
    else:
        print("\n[Matcher test skipped — no --feat-model provided]")

    test_inference_with_params(det_model, args.width, args.height)

    separator("All tests completed")


if __name__ == "__main__":
    main()
