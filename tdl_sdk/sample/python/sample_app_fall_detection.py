#!/usr/bin/env python3
"""
sample_app_fall_detection.py — Detecção de quedas + servidor RTSP

Suporta três modos de entrada:
  câmera VI  (padrão)    — sensor ligado diretamente ao SoC
  RTSP/VDEC (--input)    — stream RTSP decodificado por hardware (VDEC)
  USB/V4L2  (--input)    — câmera USB via V4L2 (UVC, /dev/videoN)

Executa detecção de pose (KEYPOINT_YOLOV8POSE_PERSON17), rastreamento
multi-pessoa e o algoritmo de detecção de quedas da versão C++
(fall_detection.cpp). Transmite o resultado anotado via RTSP e grava
resultados em disco.

Uso (câmera VI — model type auto-detectado):
    python3 sample_app_fall_detection.py \\
        --model /root/cv181x/keypoint_yolov8pose_person17_...cvimodel

Uso (câmera VI — parâmetros explícitos):
    python3 sample_app_fall_detection.py \\
        --model /root/cv181x/keypoint_yolov8pose_person17_...cvimodel \\
        --model-type KEYPOINT_YOLOV8POSE_PERSON17 \\
        [--output /tmp/fall_output] \\
        [--width 1280] [--height 720] [--fps 25]

Uso (entrada RTSP — resolução auto-detectada):
    python3 sample_app_fall_detection.py \\
        --model ... --input rtsp://192.168.1.10:554/live

Uso (câmera USB):
    python3 sample_app_fall_detection.py \\
        --model ... --input usb

Conectar ao stream de saída:
    vlc rtsp://<ip>:554/<session>
    ffplay rtsp://<ip>:554/<session>
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections import deque

import tdl
from tdl import image, nn


# ─── Auto-detecção de modelo e resolução ─────────────────────────────────────

def _detect_model_type(model_path):
    """Infere o ModelType a partir do nome do arquivo .cvimodel."""
    import os
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


# ─── Constantes do algoritmo de detecção de quedas ───────────────────────────
# Valores idênticos ao C++ (fall_detection.hpp / fall_detection.cpp)
_SCORE_THRESHOLD       = 0.4    # confiança mínima por keypoint
_FRAME_GAP             = 1      # gap de índice no histórico de posições
_SPEED_THRESHOLD       = 95.0
_HUMAN_ANGLE_THRESHOLD = 25.0
_ASPECT_RATIO_THRESHOLD = 0.6
_MAX_UNMATCHED_TIME    = 30

# Estado do tracker: NEW=0, TRACKED=1, LOST=2, REMOVED=3  (TrackStatus enum)
_TRACK_STATUS_NEW = 0


# ─── FallDet: estado de queda por pessoa ──────────────────────────────────────

class FallDet:
    """
    Porta Python directa do algoritmo C++ FallDet (fall_detection.cpp/.hpp).

    Histórico de posições:
        history_neck / history_hip — lista de até FRAME_GAP+3=4 entradas (x,y).
        Cada entrada é adicionada a cada frame, independentemente de os
        keypoints serem fiáveis ou não (ponto (0,0) quando inválidos).

    Filas circulares:
        valid_list    (maxlen=4)  — 1 se keypoints fiáveis, 0 caso contrário.
        speed_caches  (maxlen=3)  — 1 se velocidade > limiar, 0 caso contrário.
        statuses_cache(maxlen=6)  — 1 se queda detectada no frame, 0 caso contrário.

    detect() retorna 1 se a pessoa está a cair, 0 caso contrário.
    """

    def __init__(self, uid: int):
        self.uid            = uid
        self.unmatched_times = 0
        self.valid_list      = deque([0, 0, 0, 0],       maxlen=4)
        self.speed_caches    = deque([0, 0, 0],           maxlen=3)
        self.statuses_cache  = deque([0, 0, 0, 0, 0, 0], maxlen=6)
        self.history_neck: list = []
        self.history_hip:  list = []
        self.is_moving = False

    # ── Utilitários ──────────────────────────────────────────────────────────

    @staticmethod
    def _elem_count(q: deque) -> int:
        return sum(q)

    def _get_kps(self, history: list, index: int):
        """Média de 3 entradas consecutivas a partir de 'index' → (x, y)."""
        x = (history[index][0] + history[index+1][0] + history[index+2][0]) / 3.0
        y = (history[index][1] + history[index+1][1] + history[index+2][1]) / 3.0
        return x, y

    # ── Extracção de keypoints ────────────────────────────────────────────────

    def _keypoints_useful(self, lm_x, lm_y, lm_score) -> bool:
        """
        Atualiza o histórico e retorna True se os keypoints chave
        (ombros 5,6 e ancas 11,12) têm confiança suficiente.
        """
        if len(self.history_neck) == _FRAME_GAP + 3:
            self.history_neck.pop(0)
            self.history_hip.pop(0)

        if (lm_score[5]  > _SCORE_THRESHOLD and
                lm_score[6]  > _SCORE_THRESHOLD and
                lm_score[11] > _SCORE_THRESHOLD and
                lm_score[12] > _SCORE_THRESHOLD):
            neck_x = (lm_x[5]  + lm_x[6])  / 2.0
            neck_y = (lm_y[5]  + lm_y[6])  / 2.0
            hip_x  = (lm_x[11] + lm_x[12]) / 2.0
            hip_y  = (lm_y[11] + lm_y[12]) / 2.0
            self.history_neck.append((neck_x, neck_y))
            self.history_hip.append((hip_x,  hip_y))
            return True
        else:
            self.history_neck.append((0.0, 0.0))
            self.history_hip.append((0.0, 0.0))
            return False

    # ── Orientação corporal ───────────────────────────────────────────────────

    def _human_orientation(self) -> float:
        neck_x, neck_y = self._get_kps(self.history_neck, _FRAME_GAP)
        hip_x,  hip_y  = self._get_kps(self.history_hip,  _FRAME_GAP)
        angle = math.atan2(hip_y - neck_y, hip_x - neck_x) * 180.0 / math.pi - 90.0
        return angle

    # ── Proporção do bounding box ─────────────────────────────────────────────

    @staticmethod
    def _body_box_calculation(x1, y1, x2, y2) -> float:
        return (x2 - x1) / max(y2 - y1, 1e-6)

    # ── Velocidade de queda ───────────────────────────────────────────────────

    def _speed_detection(self, x1, y1, x2, y2,
                         lm_x, lm_y, lm_score, fps: float) -> float:
        neck_x_before, neck_y_before = self._get_kps(self.history_neck, 0)
        neck_x_cur,    neck_y_cur    = self._get_kps(self.history_neck, _FRAME_GAP)

        delta_pos = math.sqrt((neck_x_before - neck_x_cur)**2 +
                              (neck_y_before - neck_y_cur)**2)
        if neck_y_cur < neck_y_before:
            delta_pos = -delta_pos

        box_w = x2 - x1
        box_h = y2 - y1

        # Ajusta altura da caixa conforme visibilidade dos pés (tornozelos)
        if (lm_score[13] < _SCORE_THRESHOLD and
                lm_score[14] < _SCORE_THRESHOLD and
                lm_score[15] < _SCORE_THRESHOLD and
                lm_score[16] < _SCORE_THRESHOLD):
            box_h *= 1.8
        elif (lm_score[15] < _SCORE_THRESHOLD and
              lm_score[16] < _SCORE_THRESHOLD):
            box_h *= 1.3

        delta_body = math.sqrt(box_w**2 + box_h**2)
        delta_val  = [delta_body]

        # Perna esquerda: anca(12) → joelho(14) → tornozelo(16)
        if (lm_score[12] > _SCORE_THRESHOLD and
                lm_score[14] > _SCORE_THRESHOLD and
                lm_score[16] > _SCORE_THRESHOLD):
            ll_up  = math.hypot(lm_x[12]-lm_x[14], lm_y[12]-lm_y[14])
            ll_bot = math.hypot(lm_x[16]-lm_x[14], lm_y[16]-lm_y[14])
            delta_val.append((ll_up + ll_bot) * 2.4)

        # Perna direita: anca(11) → joelho(13) → tornozelo(15)
        if (lm_score[11] > _SCORE_THRESHOLD and
                lm_score[13] > _SCORE_THRESHOLD and
                lm_score[15] > _SCORE_THRESHOLD):
            rl_up  = math.hypot(lm_x[11]-lm_x[13], lm_y[11]-lm_y[13])
            rl_bot = math.hypot(lm_x[13]-lm_x[15], lm_y[13]-lm_y[15])
            delta_val.append((rl_up + rl_bot) * 2.4)

        # Braço esquerdo: ombro(6) → cotovelo(8) → pulso(10)
        if (lm_score[6] > _SCORE_THRESHOLD and
                lm_score[8] > _SCORE_THRESHOLD and
                lm_score[10] > _SCORE_THRESHOLD):
            la_up  = math.hypot(lm_x[6]-lm_x[8],  lm_y[6]-lm_y[8])
            la_bot = math.hypot(lm_x[8]-lm_x[10], lm_y[8]-lm_y[10])
            delta_val.append((la_up + la_bot) * 3.4)

        # Braço direito: ombro(5) → cotovelo(7) → pulso(9)
        if (lm_score[5] > _SCORE_THRESHOLD and
                lm_score[7] > _SCORE_THRESHOLD and
                lm_score[9] > _SCORE_THRESHOLD):
            ra_up  = math.hypot(lm_x[5]-lm_x[7], lm_y[5]-lm_y[7])
            ra_bot = math.hypot(lm_x[7]-lm_x[9], lm_y[7]-lm_y[9])
            delta_val.append((ra_up + ra_bot) * 3.4)

        delta_mean = sum(delta_val) / len(delta_val)
        speed = 100.0 * delta_pos / (delta_mean * (_FRAME_GAP / fps))

        self.speed_caches.append(1 if speed > _SPEED_THRESHOLD else 0)
        self.is_moving = self._elem_count(self.speed_caches) >= 2
        return speed

    # ── Análise da acção ──────────────────────────────────────────────────────

    def _action_analysis(self, human_angle: float, aspect_ratio: float,
                         moving_speed: float) -> int:
        """
        Retorna o estado mais provável:
          0=Stand_still, 1=Stand_walking, 2=Fall, 3=Lie, 4=Sit
        """
        score = [0.0] * 5

        if -_HUMAN_ANGLE_THRESHOLD < human_angle < _HUMAN_ANGLE_THRESHOLD:
            score[0] += 0.8; score[1] += 0.8; score[4] += 0.8
        else:
            score[2] += 0.8; score[3] += 0.8

        if aspect_ratio < _ASPECT_RATIO_THRESHOLD:
            score[0] += 0.8; score[1] += 0.8
        elif aspect_ratio > 1.0 / _ASPECT_RATIO_THRESHOLD:
            score[3] += 0.8
        else:
            score[2] += 0.8; score[4] += 0.8

        if moving_speed < _SPEED_THRESHOLD:
            score[0] += 0.8; score[1] += 0.8; score[3] += 0.8; score[4] += 0.8
        else:
            score[2] += 0.8

        if self.is_moving:
            score[1] += 0.8; score[2] += 0.8
        else:
            score[0] += 0.8; score[3] += 0.8; score[4] += 0.8

        return score.index(max(score))

    # ── Decisão de alerta ─────────────────────────────────────────────────────

    def _alert_decision(self, status: int) -> bool:
        self.statuses_cache.append(status)
        return self._elem_count(self.statuses_cache) >= 3

    # ── Interface pública ─────────────────────────────────────────────────────

    _ACTION_NAMES = ["Stand_still", "Stand_walking", "Fall", "Lie", "Sit"]
    # Keypoints verificados em cada camada (índices COCO-17)
    _KP_NAMES = {5: "L-shldr", 6: "R-shldr", 7: "L-elbow", 8: "R-elbow",
                 9: "L-wrist", 10: "R-wrist", 11: "L-hip", 12: "R-hip",
                 13: "L-knee", 14: "R-knee", 15: "L-ankle", 16: "R-ankle"}

    def detect(self, x1: float, y1: float, x2: float, y2: float,
               lm_x: list, lm_y: list, lm_score: list,
               fps: float) -> tuple:
        """
        Retorna (falling: int, stage: dict).

        stage contém TODOS os critérios de TODAS as camadas, sempre —
        independente de qualquer camada ter passado ou não.

        Camada 1 — keypoints:
          kp_scores  dict  {idx: score}  scores dos 4 KPs críticos (5,6,11,12)
          kp_pass    dict  {idx: bool}   se cada um passa o limiar 0.4
          kp_valid   bool  — todos os 4 passam
          valid_sum  int   — nº de frames com kp_valid=True nos últimos 4

        Camada 2 — análise de postura (calculada se history >= 4 entradas):
          has_history  bool   — histórico suficiente para cálculo
          angle        float  — ângulo corporal (graus)
          angle_pass   bool   — |angle| < 25°  (critério: não vertical)
          aspect       float  — largura/altura do bbox
          aspect_pass  str    — "standing"(<0.6) / "ambiguous" / "lying"(>1.67)
          speed        float  — velocidade de queda normalizada
          speed_pass   bool   — speed > 95
          is_moving    bool   — ≥2 dos últimos 3 frames com speed>95
          action       int    — estado classificado (0-4)
          action_name  str    — nome do estado

        Camada 3 — confirmação temporal:
          alerts       int    — nº de "Fall" nos últimos 6 frames (0-6)
          alerts_pass  bool   — alerts >= 3

        Resultado:
          falling      int    — 1 se todas as camadas passaram, 0 caso contrário
        """
        # ── Camada 1: keypoints ───────────────────────────────────────────────
        key_kps = [5, 6, 11, 12]
        kp_scores = {i: round(lm_score[i], 3) for i in key_kps}
        kp_pass   = {i: lm_score[i] > _SCORE_THRESHOLD for i in key_kps}
        kp_valid  = all(kp_pass.values())

        # Atualiza histórico (sempre, com (0,0) se inválido)
        self._keypoints_useful(lm_x, lm_y, lm_score)
        self.valid_list.append(1 if kp_valid else 0)
        valid_sum = self._elem_count(self.valid_list)
        hist_len  = len(self.history_neck)   # tamanho real do histórico (0-4)

        # ── Camada 2 + Camada 3 ──────────────────────────────────────────────
        # _speed_detection atualiza speed_caches e is_moving — só deve ser
        # chamada quando o histórico contém entradas válidas suficientes para
        # não corromper esses buffers com valores (0,0).
        #
        # Condição: has_history (hist>=4) E valid_sum>=2.
        #   - valid_sum>=2 (em vez do ==4 do C++) aceita até 2 frames com
        #     keypoints fracos nos últimos 4, o que é comum na câmera do Duo
        #     durante a transição de queda (ombros/quadris perdem confiança
        #     por 1-2 frames) sem corromper os buffers com apenas zeros.
        #   - alert_decision ainda exige valid_sum>=2 E action==2: o sinal
        #     de queda tem que ser repetido em >=3 dos últimos 6 frames.
        has_history = hist_len >= _FRAME_GAP + 3  # >= 4
        _MIN_VALID = 2  # mínimo de frames com keypoints válidos nos últimos 4

        angle = aspect = speed = action = action_name = None
        angle_pass = aspect_pass = speed_pass = None
        falling = 0

        if has_history and valid_sum >= _MIN_VALID:
            angle  = round(self._human_orientation(), 2)
            aspect = round(self._body_box_calculation(x1, y1, x2, y2), 3)
            speed  = round(self._speed_detection(x1, y1, x2, y2,
                                                 lm_x, lm_y, lm_score, fps), 2)
            status      = self._action_analysis(angle, aspect, speed)
            action      = status
            action_name = self._ACTION_NAMES[status]
            angle_pass  = abs(angle) >= _HUMAN_ANGLE_THRESHOLD
            aspect_pass = ("lying"    if aspect > 1.0 / _ASPECT_RATIO_THRESHOLD
                           else "ambiguous" if aspect >= _ASPECT_RATIO_THRESHOLD
                           else "standing")
            speed_pass  = speed > _SPEED_THRESHOLD

            is_fall = 1 if action == 2 else 0
            if self._alert_decision(is_fall):
                falling = 1

        alerts      = self._elem_count(self.statuses_cache)
        alerts_pass = alerts >= 3

        stage = dict(
            # camada 1
            kp_scores=kp_scores, kp_pass=kp_pass,
            kp_valid=kp_valid,   valid_sum=valid_sum,
            # camada 2
            hist_len=hist_len, has_history=has_history,
            angle=angle,   angle_pass=angle_pass,
            aspect=aspect, aspect_pass=aspect_pass,
            speed=speed,   speed_pass=speed_pass,
            is_moving=self.is_moving,
            action=action, action_name=action_name,
            # camada 3
            alerts=alerts, alerts_pass=alerts_pass,
            # resultado
            falling=falling,
        )
        return falling, stage


# ─── Sinalização de encerramento ──────────────────────────────────────────────

_running = True


def _sigint_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# ─── Estado partilhado entre threads ─────────────────────────────────────────

_release_queue   = []       # lista de (frame, frame_id, do_infer)
_release_lock    = threading.Lock()
_last_results    = []       # lista de dicts com bbox + fall status + landmarks
_results_lock    = threading.Lock()
_infer_fps       = 0.0
_infer_ms_acc    = 0.0      # acumulador para média por janela de reporte
_infer_count_window = 0     # contagem de inferências na janela actual
_frame_count     = 0
_fallen_ids      = set()    # track_ids que tiveram queda desde o último report
_fallen_lock     = threading.Lock()

_VB_BUFFER_NUM   = 5
_vb_sem          = threading.Semaphore(_VB_BUFFER_NUM - 2)
_MAX_RELEASE_PENDING = 2
_source_type     = "vi"   # "vi" | "vdec" | "usb"


# ─── Thread de inferência / rastreamento / detecção de quedas ────────────────

def _inference_worker(detector, cam, tracker, fps: float, output_file: str):
    """
    Thread de inferência.

    Para cada frame marcado com do_infer=True:
      1. Executa detecção de pose (KEYPOINT_YOLOV8POSE_PERSON17).
      2. Passa as detecções ao tracker SORT.
      3. Para cada track com obj_idx válido, executa FallDet.detect().
      4. Actualiza _last_results para o loop principal desenhar.
      5. Grava resultado em disco (se output_dir fornecido).

    Para todos os frames (do_infer ou não):
      - Chama cam.release() em ordem FIFO (vi) ou release_inference() (vdec).
      - Libera o semáforo de VB apenas para fonte "vi".
    """
    global _running, _infer_fps, _last_results, _source_type, _fallen_ids
    global _infer_ms_acc, _infer_count_window

    muti_person: dict = {}    # {track_id: FallDet}
    # Contador sequencial para o tracker — NÃO usa frame_id da câmera.
    # O C++ fornece frame_ids consecutivos (0,1,2,...) porque o pipeline
    # processa todos os frames. No Python, do_infer=False pula frames e o
    # frame_id da câmera salta (0,5,10,...). O SORT interpreta os saltos
    # como frames sem match e remove o track → novo track_id → FallDet reset.
    # Com infer_frame_id (incrementa a cada chamada ao tracker) o SORT vê
    # a mesma sequência contínua que vê no C++.
    infer_frame_id = 0
    count = 0
    t0 = time.time()

    while True:
        item = None
        with _release_lock:
            if _release_queue:
                item = _release_queue.pop(0)

        if item is None:
            if not _running:
                break          # fila vazia e sinal de parada: encerra
            time.sleep(0.001)
            continue

        frame, frame_id, do_infer = item

        if do_infer:
            # ── 1. Detecção de pose ─────────────────────────────────────────
            dets = detector.inference(frame)
            # Use C++ steady_clock measurement (excludes GIL wait time).
            dt = detector.get_last_inference_ms()
            _infer_ms_acc += dt
            _infer_count_window += 1

            # ── 2. Rastreamento ─────────────────────────────────────────────
            # O modelo YoloV8Pose não chama setTypeMapping(), portanto
            # object_type fica OBJECT_TYPE_UNDEFINED. O tracker SORT filtra
            # boxes UNDEFINED (mot.cpp:38) e nunca as associa a tracks
            # existentes → novo track_id a cada frame → hist_len fica em 1.
            # No C++ o getTrackNode() corrige explicitamente:
            #   box_info.object_type = OBJECT_TYPE_PERSON; (fall_detection_app.cpp:270)
            # Fazemos o mesmo aqui antes de chamar tracker.track().
            for d in dets:
                d["class_name"] = "PERSON"
            tracker.set_img_size(frame.get_size()[0], frame.get_size()[1])
            track_results = tracker.track(dets, infer_frame_id) if dets else []
            infer_frame_id += 1

            # ── 3. Detecção de quedas ────────────────────────────────────────
            # Monta mapa completo de tracks activos neste frame.
            # Não distingue NEW vs TRACKED aqui: se o track_id já existe em
            # muti_person ele é SEMPRE reutilizado. Isso evita o bug em que o
            # Python SORT reporta status=NEW em frames consecutivos para o mesmo
            # track_id, causando recriação do FallDet e reset do histórico.
            active_index = {}   # {track_id: obj_idx}  — todos os tracks visíveis
            for t in track_results:
                if t["obj_idx"] != -1:
                    active_index[t["track_id"]] = t["obj_idx"]

            det_results = {}   # {track_id: (falling, stage)}
            to_remove   = []

            # Actualiza tracks já conhecidos
            for tid, fall_det in muti_person.items():
                if tid in active_index:
                    idx = active_index[tid]
                    d   = dets[idx]
                    lm   = d.get("landmarks", [])
                    lm_x = [pt[0] for pt in lm]
                    lm_y = [pt[1] for pt in lm]
                    lm_s = d.get("landmarks_score", [1.0] * len(lm))
                    if not isinstance(lm_s, (list, tuple)):
                        lm_s = [lm_s] * len(lm)
                    det_results[tid] = fall_det.detect(
                        d["x1"], d["y1"], d["x2"], d["y2"],
                        lm_x, lm_y, lm_s, fps)
                    fall_det.unmatched_times = 0
                else:
                    fall_det.unmatched_times += 1
                    fall_det.valid_list.append(0)
                    if fall_det.unmatched_times >= _MAX_UNMATCHED_TIME:
                        to_remove.append(tid)
                    else:
                        det_results[tid] = (0, None)
            for tid in to_remove:
                del muti_person[tid]

            # Cria FallDet apenas para track_ids genuinamente novos
            for tid, idx in active_index.items():
                if tid in muti_person:
                    continue   # já foi processado acima
                d   = dets[idx]
                lm   = d.get("landmarks", [])
                lm_x = [pt[0] for pt in lm]
                lm_y = [pt[1] for pt in lm]
                lm_s = d.get("landmarks_score", [1.0] * len(lm))
                if not isinstance(lm_s, (list, tuple)):
                    lm_s = [lm_s] * len(lm)
                fd = FallDet(tid)
                muti_person[tid] = fd
                det_results[tid] = fd.detect(
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    lm_x, lm_y, lm_s, fps)

            # ── 4. Prepara resultados para o loop principal ──────────────────
            results = []
            for t in track_results:
                if t["obj_idx"] == -1:
                    continue
                tid  = t["track_id"]
                idx  = t["obj_idx"]
                d    = dets[idx]
                bi   = t["box_info"]
                pair = det_results.get(tid, (0, None))
                falling_val, stage = pair
                results.append({
                    "track_id":        tid,
                    "x1":              bi["x1"],
                    "y1":              bi["y1"],
                    "x2":              bi["x2"],
                    "y2":              bi["y2"],
                    "score":           bi["score"],
                    "falling":         bool(falling_val),
                    "stage":           stage,
                    "landmarks":       d.get("landmarks", []),
                    "landmarks_score": d.get("landmarks_score", []),
                })

            with _results_lock:
                _last_results = results

            # Registra track_ids com queda ativa para o relatório do terminal
            falling_now = {r["track_id"] for r in results if r["falling"]}
            if falling_now:
                with _fallen_lock:
                    _fallen_ids.update(falling_now)

            # ── 5. Saída em disco ────────────────────────────────────────────
            if output_file:
                _write_result(output_file, frame_id, results,
                              frame.get_size()[0], frame.get_size()[1])

            count += 1
            total  = time.time() - t0
            if total > 0:
                _infer_fps = count / total

        # Libera sempre em ordem FIFO.
        # vi:   cam.release() devolve buffer ao VB pool + libera semáforo.
        # vdec: release_inference() libera o slot de inferência
        #         (slot de display já liberado pelo loop após pin_for_inference).
        # usb:  cam.release() é no-op; VPSSImage gerencia própria memória ION.
        if _source_type == "vi":
            cam.release()
            _vb_sem.release()
        elif _source_type == "vdec":
            cam.release_inference()
        else:
            cam.release()


def _write_result(output_file: str, frame_id: int, results: list,
                  img_w: int, img_h: int):
    """
    Acrescenta ao arquivo os critérios de TODAS as camadas para cada pessoa,
    independente de qualquer camada ter passado. Formato:

    --- frame=00001234  track=1 ---
    [C1] kp_valid=N  valid=2/4
         kp5=L-shldr  score=0.312  FAIL(<0.4)
         kp6=R-shldr  score=0.451  ok
         kp11=L-hip   score=0.289  FAIL(<0.4)
         kp12=R-hip   score=0.521  ok
    [C2] has_history=Y
         angle=+72.3  (>=25 -> fall crit OK)
         aspect=1.124  (ambiguous -> fall crit OK)
         speed=112.7   (>=95 -> fall crit OK)
         is_moving=Y
         action=Fall(2)
    [C3] alerts=2/6  (need>=3)
    => ok
    """
    if not results:
        return

    _KP_NAMES = FallDet._KP_NAMES

    lines = []
    for r in results:
        s = r.get("stage")
        tid = int(r["track_id"])

        lines.append(f"--- frame={frame_id:08d}  track={tid} ---\n")

        if s is None:
            lines.append("    [unmatched]\n")
            continue

        # ── Camada 1 ──────────────────────────────────────────────────────────
        kp_tag = "ok  " if s["kp_valid"] else "FAIL"
        lines.append(f"[C1] kp_valid={kp_tag}  valid={s['valid_sum']}/4\n")
        for idx in [5, 6, 11, 12]:
            sc   = s["kp_scores"][idx]
            ok   = s["kp_pass"][idx]
            tag  = "ok" if ok else f"FAIL(<{_SCORE_THRESHOLD})"
            name = _KP_NAMES.get(idx, f"kp{idx}")
            lines.append(f"     kp{idx}={name:<9s}  score={sc:.3f}  {tag}\n")

        # ── Camada 2 ──────────────────────────────────────────────────────────
        # hist_len = tamanho real do histórico (cresce 0→4 independente de kp_valid)
        # valid_sum = quantos dos últimos 4 frames tiveram kp_valid=True
        # has_history = hist_len >= 4  (precisa de 4 para _get_kps funcionar)
        hl = s["hist_len"]
        if s["has_history"]:
            hist_tag = f"Y  (hist={hl}/4  valid_frames={s['valid_sum']}/4)"
        else:
            hist_tag = (f"N  hist={hl}/4 — precisa de 4 entradas no histórico; "
                        f"valid_frames={s['valid_sum']}/4  "
                        f"(se hist fica em 1: track_id muda a cada frame)")
        lines.append(f"[C2] has_history={hist_tag}\n")

        if s["angle"] is not None:
            # angle: queda → |angle| >= 25
            a_tag = f"{'OK ' if s['angle_pass'] else 'no '}  (|angle|>={_HUMAN_ANGLE_THRESHOLD}->fall)"
            lines.append(f"     angle  ={s['angle']:+7.2f}°  {a_tag}\n")

            # aspect: queda → ambiguous (0.6..1.67)
            a2_tag = f"{s['aspect_pass']:<9s}  (standing<{_ASPECT_RATIO_THRESHOLD} / ambiguous->fall / lying>{1/_ASPECT_RATIO_THRESHOLD:.2f})"
            lines.append(f"     aspect ={s['aspect']:8.3f}   {a2_tag}\n")

            # speed
            sp_tag = f"{'OK ' if s['speed_pass'] else 'no '}  (>={_SPEED_THRESHOLD}->fall)"
            lines.append(f"     speed  ={s['speed']:8.2f}   {sp_tag}\n")

            lines.append(f"     is_moving={'Y' if s['is_moving'] else 'N'}\n")
            lines.append(f"     action = {s['action_name']}({s['action']})\n")
        elif not s["has_history"]:
            lines.append(f"     (skipped — histórico insuficiente: {s['hist_len']}/4 frames)\n")
        else:
            lines.append(f"     (skipped — kps insuficientes: valid={s['valid_sum']}/4, "
                         f"precisa >={2}; verifique scores dos kp5,6,11,12)\n")

        # ── Camada 3 ──────────────────────────────────────────────────────────
        al_tag = "OK " if s["alerts_pass"] else "no "
        lines.append(f"[C3] alerts={s['alerts']}/6  {al_tag}(need>=3)\n")

        # Resultado
        lines.append(f"=> {'FALL' if s['falling'] else 'ok  '}\n")
        lines.append("\n")

    try:
        with open(output_file, "a") as f:
            f.writelines(lines)
    except OSError as e:
        print(f"[AVISO] Não foi possível gravar resultado: {e}")


# ─── Desenho de overlay ───────────────────────────────────────────────────────

def _draw_fall_overlay(frame, results: list, threshold: float):
    """
    Desenha bounding boxes e indicadores de queda sobre o frame.
      - Verde + ID: pessoa normal
      - Vermelho + "FALL!": queda detectada
    """
    if not results:
        return

    # Keypoints: passa dets no formato esperado por draw_keypoints
    kp_dets = []
    for r in results:
        if r.get("landmarks"):
            kp_dets.append({
                "x1":              r["x1"],
                "y1":              r["y1"],
                "x2":              r["x2"],
                "y2":              r["y2"],
                "score":           r["score"],
                "class_id":        0,
                "class_name":      "PERSON",
                "landmarks":       r["landmarks"],
                "landmarks_score": r.get("landmarks_score", []),
            })
    if kp_dets:
        image.draw_keypoints(frame, kp_dets, score_threshold=threshold)

    # Bounding boxes + labels
    for r in results:
        falling = r["falling"]
        x1, y1  = int(r["x1"]), int(r["y1"])
        x2, y2  = int(r["x2"]), int(r["y2"])
        tid     = r["track_id"]

        if falling:
            color = (255, 0, 0)   # vermelho (R,G,B)
            label = f"FALL! id={tid}"
        else:
            color = (0, 255, 0)   # verde
            label = f"id={tid}"

        image.draw_bbox(frame, x1, y1, x2, y2, color=color, thickness=2)
        image.draw_text(frame, label, x1, max(0, y1 - 4),
                        color=color, scale=0.5)


# ─── Argumentos ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detecção de quedas com câmera VI ou entrada RTSP + servidor RTSP")
    p.add_argument("--model", required=True,
                   help="Caminho para o .cvimodel de pose (KEYPOINT_YOLOV8POSE_PERSON17)")
    p.add_argument("--model-type", default="", dest="model_type",
                   help="ModelType do modelo de pose (ex: KEYPOINT_YOLOV8POSE_PERSON17). "
                        "Auto-detectado pelo nome do arquivo se omitido.")
    p.add_argument("--input", default="",
                   help="Fonte de vídeo: vazio = câmera VI local; "
                        "rtsp://... = stream RTSP (VDEC hardware); "
                        "usb = câmera USB /dev/video0; usb:1 = /dev/video1.")
    p.add_argument("--transport", default="tcp", choices=["tcp", "udp"],
                   help="Protocolo de transporte RTSP de entrada (padrão: tcp)")
    p.add_argument("--output", default="",
                   help="Arquivo .txt de saída para registrar quedas detectadas (opcional). "
                        "Cada linha corresponde a uma queda; o arquivo é criado/sobrescrito "
                        "ao iniciar e os eventos são acrescentados durante a execução. "
                        "Se omitido, apenas transmite via RTSP.")
    p.add_argument("--width",   type=int, default=0,
                   help="Largura em pixels (auto-detectado se omitido; "
                        "padrão: 1280 para VI, 640 para USB)")
    p.add_argument("--height",  type=int, default=0,
                   help="Altura em pixels (auto-detectado se omitido; "
                        "padrão: 720 para VI, 480 para USB)")
    p.add_argument("--fps",     type=float, default=25.0,
                   help="FPS da câmera/pipeline (usado no algoritmo de velocidade, padrão: 25)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Limiar de confiança da detecção (padrão: 0.5)")
    p.add_argument("--codec",   default="h264", choices=["h264", "h265"])
    p.add_argument("--bitrate", type=int, default=3072,
                   help="Bitrate RTSP em kbps (padrão: 3072)")
    p.add_argument("--gop",     type=int, default=0,
                   help="Intervalo de keyframe em frames (padrão: 0 = automático: 1× fps)")
    p.add_argument("--session", default="live",
                   help="Nome da sessão RTSP de saída (padrão: live)")
    p.add_argument("--mirror",  action="store_true", default=False,
                   help="Espelho horizontal (apenas câmera VI)")
    p.add_argument("--flip",    action="store_true", default=False,
                   help="Flip vertical (apenas câmera VI)")
    p.add_argument("--frames",  type=int, default=0,
                   help="Número de frames a processar; 0 = infinito (padrão)")
    return p.parse_args()


# ─── Abertura da fonte de vídeo ───────────────────────────────────────────────

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
        cam = image.Camera(args.width, args.height,
                           image.ImageFormat.YUV420SP_VU,
                           vb_buffer_num=_VB_BUFFER_NUM,
                           mirror=args.mirror, flip=args.flip)
        print("  Backend: câmera VI local")
        return cam, "vi"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _running, _frame_count, _source_type
    global _infer_ms_acc, _infer_count_window

    args = parse_args()

    # --- Arquivo de saída (criado/truncado no início para não acumular runs anteriores) ---
    if args.output:
        try:
            open(args.output, "w").close()   # cria ou limpa o arquivo
        except OSError as e:
            print(f"[ERRO] Não foi possível criar o arquivo de saída '{args.output}': {e}")
            sys.exit(1)
        print(f"Quedas registradas em: {args.output}")

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

    # --- Modelo de pose ---
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

    # --- Tracker SORT ---
    tracker = nn.Tracker(nn.TrackerType.MOT_SORT)

    # --- Fonte de vídeo ---
    cam, _source_type = _open_source(args)

    # --- Auto-ajuste fps/gop para câmera USB ---
    rtsp_fps = int(args.fps)
    rtsp_bitrate = args.bitrate
    if _source_type == "usb" and rtsp_fps > 10:
        rtsp_fps = 3
        print(f"  [auto] FPS ajustado para {rtsp_fps} (câmera USB geralmente entrega ≤5fps)")
    if _source_type == "usb" and rtsp_bitrate >= 2048:
        rtsp_bitrate = 1024
        print(f"  [auto] Bitrate ajustado para {rtsp_bitrate}kbps (USB: NALUs menores → "
              f"compatível com VLC/UDP)")
    rtsp_gop = args.gop if args.gop > 0 else max(1, rtsp_fps)
    if _source_type == "usb" and args.gop <= 0:
        rtsp_gop = max(1, rtsp_fps)
        print(f"  [auto] GOP ajustado para {rtsp_gop} (1 keyframe/s para câmera USB)")

    # --- Servidor RTSP de saída ---
    print(f"\nIniciando RTSP {args.width}x{args.height} "
          f"codec={args.codec} bitrate={rtsp_bitrate}kbps "
          f"fps={rtsp_fps} gop={rtsp_gop} ({rtsp_gop/rtsp_fps:.1f}s) "
          f"sessão={args.session} ...")
    rtsp = image.RTSPServer(
        args.width, args.height,
        chn=0,
        codec=args.codec,
        session_name=args.session,
        bitrate=rtsp_bitrate,
        gop=rtsp_gop,
        fps=rtsp_fps,
    )
    print(f"  Stream: rtsp://<ip>:554/{args.session}")
    print(f"  VLC: vlc --rtsp-tcp rtsp://<ip>:554/{args.session}")

    # --- Thread de inferência ---
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(detector, cam, tracker, args.fps, args.output),
        daemon=True,
    )
    infer_thread.start()

    # --- Loop principal ---
    frame_idx  = 0
    _frame_limit = args.frames if args.frames > 0 else None
    t_start    = time.time()
    t_report   = t_start

    print("\nDetectando quedas... (Ctrl+C para parar)\n")

    try:
        while _running and (_frame_limit is None or frame_idx < _frame_limit):
            if _source_type == "vi" and not _vb_sem.acquire(timeout=0.5):
                continue

            frame = cam.read()

            # Lê últimos resultados e desenha overlay
            with _results_lock:
                results = list(_last_results)

            _draw_fall_overlay(frame, results, args.threshold)
            rtsp.send_frame(frame)

            # Para RtspClientVdec: move o frame para o slot de inferência e
            # libera o slot de display imediatamente, permitindo o próximo read().
            # A thread de inferência usa o frame e chama release_inference().
            if _source_type == "vdec":
                cam.pin_for_inference()

            do_infer = True
            with _release_lock:
                if len(_release_queue) >= _MAX_RELEASE_PENDING:
                    do_infer = False
                # USB: cam.release() é no-op, então frames sem inferência
                # não precisam da fila — VB blocks liberados pelo GC quando
                # a referência local 'frame' é sobrescrita no próximo read().
                if do_infer or _source_type != "usb":
                    _release_queue.append((frame, frame_idx, do_infer))

            if _source_type == "vdec":
                cam.release()  # libera slot de display; slot de inferência ainda vivo
            frame_idx    += 1
            _frame_count  = frame_idx

            # Relatório a cada 5 s
            now = time.time()
            if now - t_report >= 5.0:
                elapsed = now - t_start
                cam_fps = frame_idx / elapsed if elapsed > 0 else 0
                ni = max(_infer_count_window, 1)
                avg_infer = _infer_ms_acc / ni
                with _fallen_lock:
                    n_fall = len(_fallen_ids)
                    _fallen_ids.clear()
                print(f"  frame {frame_idx:6d}  "
                      f"cam={cam_fps:5.1f}fps  "
                      f"inf={_infer_fps:4.1f}fps  "
                      f"inf={avg_infer:5.1f}ms  "
                      f"pessoas={len(results)}  quedas={n_fall}")
                _infer_ms_acc = 0.0
                _infer_count_window = 0
                t_report = now

    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        import traceback; traceback.print_exc()
    finally:
        _running = False
        if _source_type == "vi":
            _vb_sem.release()
        infer_thread.join(timeout=3.0)
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


if __name__ == "__main__":
    main()
