"""video_inference — engine de inferência + writer JSONL para o DVR.

Usado por sample_dvr.py (e outros samples que queiram AI overlay).

Arquitetura:
  - Execução SÍNCRONA no record loop (quem chama decide pacing).
    Em CV181X single-core thread separado não traz paralelismo real:
    o TPU é hardware, o CPU fica bloqueado pelo próprio GIL durante
    pré/pós-processamento. Pacing oportunístico no caller (só chama
    quando tem folga no loop) consegue o mesmo efeito do async com
    muito menos complexidade.

  - Sidecar JSONL por segmento: <stem>.jsonl ao lado de <stem>.mp4.
    Uma linha JSON por frame com detecções. Bboxes normalizados 0-1.

  - Detecções "latest" protegidas por lock para leitura do SSE.

Uso típico no record loop:

    eng = VideoInferenceEngine("YOLOV8N_DET_PET_PERSON")

    for frame in ...:
        # em cada rotação de segmento:
        eng.open_segment(rec.current_segment(), rec.output_dir())
        t_ms = time_now_ms - rec.segment_start_ms()
        eng.infer(frame, t_ms, frame_w, frame_h)   # síncrono

    eng.close()
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional

from tdl import nn

_MODEL_FACTORY_PATH = "/mnt/system/configs/model/model_factory.json"
# O TDLModelFactory C++ resolve arquivos como:
#   <model_dir>/<platform>/<file_name>_<platform>.cvimodel
# Os .cvimodel ficam em /root/cv181x, logo o model_dir passado ao factory
# é o PAI (/root) — o factory completa com /cv181x/xxx_cv181x.cvimodel.
_DEFAULT_MODEL_DIR      = "/root"
_MODEL_PLATFORM_SUBDIR  = "cv181x"

# Nomes das 80 classes do dataset COCO (ordem canônica). Aplicado quando o
# model_factory.json marca `is_coco_types: true` e o C++ não preenche os
# nomes no output — evita ver "cls_3" em vez de "car" na UI.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_model_factory(path: str = _MODEL_FACTORY_PATH) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"model_list": {}}


def list_available_models(model_dir: str = _DEFAULT_MODEL_DIR,
                          factory_path: str = _MODEL_FACTORY_PATH) -> List[Dict]:
    """Retorna modelos listados no factory cujo arquivo .cvimodel existe em
    model_dir. Cada dict tem: name (chave do ModelType), file_name, types."""
    factory = load_model_factory(factory_path)
    models  = factory.get("model_list", {})
    out: List[Dict] = []
    # Os .cvimodel moram no subdir da plataforma dentro de model_dir.
    scan_dir = os.path.join(model_dir, _MODEL_PLATFORM_SUBDIR)
    if not os.path.isdir(scan_dir):
        # Aceita também caso o próprio model_dir já seja o subdir.
        scan_dir = model_dir
        if not os.path.isdir(scan_dir):
            return out
    # Lista stems de arquivos presentes (sem extensão). O SDK coloca um
    # sufixo de plataforma no arquivo (ex: _cv181x, _cv181x_v2) que o JSON
    # não contém — a comparação é por prefixo.
    present_stems = []
    try:
        for fn in os.listdir(scan_dir):
            for ext in (".cvimodel", ".bmodel"):
                if fn.endswith(ext):
                    present_stems.append(fn[:-len(ext)])
                    break
    except OSError:
        return out

    def file_available(file_name: str) -> bool:
        # Match exato ou prefixo + "_" (cobre sufixos _cv181x, _cv181x_v2 ...)
        for stem in present_stems:
            if stem == file_name or stem.startswith(file_name + "_"):
                return True
        return False

    # Modelos 2-stage esperam recorte (rosto/mão/corpo) como entrada. Rodar
    # direto no frame cheio gera detecções sem sentido. Filtrados aqui até
    # implementarmos o pipeline em 2 estágios (detector + crop + modelo).
    def is_single_stage(name: str) -> bool:
        two_stage_prefixes = (
            "KEYPOINT_FACE",       # exige crop do rosto
            "KEYPOINT_HAND",       # exige crop da mão
            "KEYPOINT_SIMCC",      # human pose (exige crop da pessoa)
            "KEYPOINT_LICENSE",    # placa (exige crop)
            "FEATURE_",            # extrator de features (exige crop)
            "CLS_ATTRIBUTE",       # atributos faciais (exige crop)
            "CLS_HAND_GESTURE",    # exige crop da mão
            "CLS_KEYPOINT",        # exige keypoints de entrada
            "CLS_RGBLIVENESS",    # liveness (exige crop do rosto)
            "CLS_MASK",            # máscara (exige crop)
            "CLS_SOUND",           # áudio, não imagem
            "RECOGNITION_",        # reconhecimento (exige crop)
        )
        return not name.startswith(two_stage_prefixes)

    # Blocklist: modelos que causam segfault ao carregar/rodar no binding
    # Python atual. Remover quando o suporte C++ for corrigido.
    BROKEN_MODELS: set = set()

    for name, meta in sorted(models.items()):
        file_name = meta.get("file_name", "")
        if not file_name or not file_available(file_name):
            continue
        if not hasattr(nn.ModelType, name):
            continue
        if not is_single_stage(name):
            continue
        if name in BROKEN_MODELS:
            continue
        out.append({
            "name":      name,
            "file_name": file_name,
            "types":     meta.get("types", []),
        })
    return out


def _name_from_path(model_path: str) -> str:
    """Dado um .cvimodel path, encontra o NAME (chave do enum) no factory
    cujo file_name bate com o stem do arquivo. Retorna '' se não achar."""
    base = os.path.basename(model_path)
    # remove extensão e sufixo de plataforma (ex: _cv181x)
    stem = base
    for ext in (".cvimodel", ".bmodel"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    factory = load_model_factory()
    for name, meta in factory.get("model_list", {}).items():
        fn = meta.get("file_name", "")
        if not fn:
            continue
        if stem == fn or stem.startswith(fn + "_"):
            return name
    return ""


def resolve_model_type(model_name: str,
                       explicit_type: Optional[str] = None):
    """Resolve string → nn.ModelType enum.

    Aceita:
      - chave do enum direto (ex: "YOLOV8_DET_COCO80")
      - caminho de arquivo .cvimodel (converte pelo factory)

    `explicit_type` (opcional) força um ModelType específico, útil quando
    o nome do modelo não bate com o enum. Normalmente não precisa.

    Levanta ValueError se não conseguir resolver.
    """
    key = explicit_type or model_name
    # Se é um path, tenta achar o nome pelo factory
    if not explicit_type and ("/" in key or key.endswith((".cvimodel",
                                                           ".bmodel"))):
        resolved = _name_from_path(key)
        if not resolved:
            raise ValueError(
                f"Não achei entrada no model_factory.json para '{key}'. "
                f"Passe o nome do enum em vez do path, ex: 'YOLOV8_DET_COCO80'.")
        key = resolved
    if not hasattr(nn.ModelType, key):
        available = sorted([a for a in dir(nn.ModelType)
                            if not a.startswith('_')])
        hint = ""
        # sugere nomes parecidos
        up = key.upper()
        matches = [a for a in available if up in a][:5]
        if matches:
            hint = f" Parecidos: {', '.join(matches)}."
        raise ValueError(f"ModelType '{key}' não existe."
                         f" Total disponível: {len(available)}.{hint}")
    return getattr(nn.ModelType, key)


class VideoInferenceEngine:
    """Carrega um modelo, roda inferência síncrona, publica detecções, escreve JSONL.

    Thread-safety: infer() é chamada pelo record thread; latest() é chamada
    por threads do servidor web (SSE). O lock protege só o snapshot publicado;
    a inferência em si é single-threaded por natureza (um TPU).
    """

    def __init__(self,
                 model_type,
                 model_dir: str = _DEFAULT_MODEL_DIR,
                 threshold: float = 0.5,
                 device_id: int = 0,
                 model_name: str = ""):
        self.model_name = model_name or str(model_type)
        # Classes vindas do factory (pode faltar em alguns modelos).
        factory  = load_model_factory()
        meta     = factory.get("model_list", {}).get(model_name, {})
        self._class_names: List[str] = list(meta.get("types", []))
        # Modelos COCO80 só marcam `is_coco_types` — usa a tabela canônica.
        if not self._class_names and meta.get("is_coco_types"):
            self._class_names = list(COCO80_NAMES)
        # Fallback quando o modelo não declara "types": heurística simples
        # baseada em tokens no nome, pra evitar label "UNDEFINED" no overlay.
        if not self._class_names:
            tokens = model_name.split("_")
            hints = {
                "PERSON":     "person",
                "PERSON17":   "person",
                "FACE":       "face",
                "HAND":       "hand",
                "LANE":       "lane",
                "VEHICLE":    "vehicle",
                "CAR":        "car",
                "BICYCLE":    "bicycle",
                "MOTOR":      "motorcycle",
                "EBICYCLE":   "ebicycle",
                "PET":        "pet",
                "HEAD":       "head",
                "HARDHAT":    "hardhat",
                "FIRE":       "fire",
                "SMOKE":      "smoke",
            }
            for t in tokens:
                if t in hints:
                    self._class_names = [hints[t]]
                    break
        self.model = nn.get_model_from_dir(model_type, model_dir,
                                           device_id=device_id)
        try:
            self.model.set_threshold(threshold)
        except Exception:
            pass
        self._lock = threading.Lock()
        # Protege contra race: troca de modelo não pode destruir o objeto
        # C++ enquanto uma inferência está em voo (segfault garantido).
        # infer() e close() disputam o mesmo lock.
        self._io_lock  = threading.Lock()
        self._closed   = False
        self._latest_t_ms: int = 0
        self._latest_dets: List[Dict] = []
        self._latest_seg: str = ""
        self._jsonl_file = None
        self._jsonl_stem: str = ""
        self._last_flush: float = 0.0

    # ── Control ─────────────────────────────────────────────────────────────
    def set_threshold(self, threshold: float) -> None:
        try:
            self.model.set_threshold(threshold)
        except Exception as e:
            print(f"[video_inference] set_threshold falhou: {e}")

    def close(self) -> None:
        # Espera qualquer inferência em voo terminar antes de destruir o
        # modelo C++ — infer() detém _io_lock durante model.inference().
        with self._io_lock:
            if self._closed:
                return
            self._closed = True
            self.close_segment()
            try:
                self.model.close()
            except Exception:
                pass

    # ── Segment lifecycle ───────────────────────────────────────────────────
    def open_segment(self, segment_basename: str, out_dir: str) -> None:
        """Abre o JSONL do segmento. Idempotente quando o basename já é o atual.
        Chamar toda vez que rec.current_segment() muda.

        Modo "a" (append) evita truncar detecções de um modelo anterior
        quando o usuário troca de modelo no meio de um segmento — nesse caso
        o arquivo já existe, preservamos o conteúdo e adicionamos um
        marcador "switch" indicando a mudança."""
        if not segment_basename:
            return
        if segment_basename == self._jsonl_stem:
            return
        self.close_segment()
        stem = segment_basename[:-4] if segment_basename.endswith(".mp4") \
               else segment_basename
        path = os.path.join(out_dir, stem + ".jsonl")
        pre_existing = os.path.exists(path) and os.path.getsize(path) > 0
        try:
            self._jsonl_file = open(path, "a")
            self._jsonl_stem = segment_basename
            if pre_existing:
                # Troca dinâmica de modelo — registra a mudança pro player
                # saber que as detecções subsequentes vieram de outro modelo.
                marker = {"type":    "switch",
                          "model":   self.model_name,
                          "segment": segment_basename}
                self._jsonl_file.write(json.dumps(marker) + "\n")
            else:
                # Primeiro write no segmento — header com metadados
                header = {"type":    "header",
                          "model":   self.model_name,
                          "segment": segment_basename}
                self._jsonl_file.write(json.dumps(header) + "\n")
            self._jsonl_file.flush()
        except OSError as e:
            print(f"[video_inference] erro abrindo {path}: {e}")
            self._jsonl_file = None
            self._jsonl_stem = ""

    def close_segment(self) -> None:
        if self._jsonl_file:
            try:
                self._jsonl_file.flush()
                self._jsonl_file.close()
            except OSError:
                pass
        self._jsonl_file = None
        self._jsonl_stem = ""

    # ── Inference ───────────────────────────────────────────────────────────
    def infer(self, frame, t_ms: int,
              frame_w: int, frame_h: int) -> List[Dict]:
        """Roda inferência no frame, publica snapshot, appenda no JSONL.

        Retorna a lista normalizada (mesma publicada em latest()).
        Bboxes são normalizados 0-1 pelo w/h do frame.

        Thread-safety: segura _io_lock durante a chamada C++ pra bloquear
        close() durante a troca dinâmica de modelo.
        """
        with self._io_lock:
            if self._closed:
                return []
            try:
                raw = self.model.inference(frame)
            except Exception as e:
                print(f"[video_inference] inference falhou: {e}")
                return []

        # Aceita só output de detecção (lista de dicts com x1,y1,x2,y2).
        if not isinstance(raw, list):
            return []

        inv_w = 1.0 / frame_w if frame_w > 0 else 1.0
        inv_h = 1.0 / frame_h if frame_h > 0 else 1.0

        def _norm_kp(pt, default_score: float = 1.0):
            """Aceita [x,y], [x,y,s] ou {'x','y','score'}; retorna [nx,ny,ns]."""
            if isinstance(pt, dict):
                x, y = pt.get("x", 0.0), pt.get("y", 0.0)
                s = pt.get("score", default_score)
            elif isinstance(pt, (list, tuple)):
                x = pt[0] if len(pt) > 0 else 0.0
                y = pt[1] if len(pt) > 1 else 0.0
                s = pt[2] if len(pt) > 2 else default_score
            else:
                return None
            return [round(float(x) * inv_w, 4),
                    round(float(y) * inv_h, 4),
                    round(float(s), 3)]

        def _extract_kps(d: Dict) -> List:
            """Extrai keypoints em formato normalizado de qualquer um dos
            layouts possíveis do SDK (landmarks+landmarks_score, keypoints,
            list simples). Retorna [] se não houver."""
            # landmarks + landmarks_score (Simcc, keypoint-only)
            if "landmarks" in d:
                lms = d.get("landmarks") or []
                scores = d.get("landmarks_score")
                if isinstance(scores, list) and len(scores) == len(lms):
                    return [
                        _norm_kp(list(lm) + [s])
                        for lm, s in zip(lms, scores)
                    ]
                # sem score por ponto (e.g. face 5-pt, yolov8-pose sem score)
                base = float(d.get("score", 1.0))
                return [_norm_kp(list(lm) + [base]) for lm in lms]
            # keypoints (lista de dicts/tuplas)
            if "keypoints" in d and isinstance(d["keypoints"], list):
                return [_norm_kp(p) for p in d["keypoints"] if _norm_kp(p)]
            return []

        dets: List[Dict] = []
        for d in raw:
            if not isinstance(d, dict):
                # Alguns modelos retornam só lista de landmarks sem bbox:
                if isinstance(d, list):
                    kps = [_norm_kp(p) for p in d if _norm_kp(p)]
                    if kps:
                        dets.append({"cls": "kps", "id": 0, "s": 1.0,
                                     "kps": [k for k in kps if k]})
                continue

            raw_cls = d.get("class_name") or ""
            class_id = int(d.get("class_id", 0))
            # Substitui class_name genérico pelo types[class_id] do factory
            # (COCO80 ou custom). O C++ devolve "UNDEFINED", "" ou "clsN" como
            # placeholder quando não tem tabela de nomes — qualquer desses
            # aciona o fallback aqui.
            is_placeholder = (
                not raw_cls
                or raw_cls == "UNDEFINED"
                or raw_cls.startswith("cls")
                and raw_cls[3:].lstrip("_").isdigit()
            )
            if is_placeholder:
                if 0 <= class_id < len(self._class_names):
                    raw_cls = self._class_names[class_id]
                elif self._class_names:
                    raw_cls = self._class_names[0]
                else:
                    raw_cls = f"cls_{class_id}"
            entry: Dict = {
                "cls": raw_cls,
                "id":  class_id,
                "s":   round(float(d.get("score", 0.0)), 3),
            }
            has_any = False
            if "x1" in d:
                entry["bb"] = [
                    round(float(d["x1"]) * inv_w, 4),
                    round(float(d["y1"]) * inv_h, 4),
                    round(float(d["x2"]) * inv_w, 4),
                    round(float(d["y2"]) * inv_h, 4),
                ]
                has_any = True
            kps = _extract_kps(d)
            kps = [k for k in kps if k]
            if kps:
                entry["kps"] = kps
                has_any = True
            if has_any:
                dets.append(entry)

        with self._lock:
            self._latest_t_ms = t_ms
            self._latest_dets = dets
            self._latest_seg  = self._jsonl_stem

        # Appenda linha no JSONL (sempre, mesmo quando vazio — útil pra
        # medir taxa de inferência e gap de tempo entre análises).
        if self._jsonl_file:
            try:
                line = json.dumps({"t": t_ms, "dets": dets}) + "\n"
                self._jsonl_file.write(line)
                now = time.time()
                if now - self._last_flush >= 1.0:
                    self._jsonl_file.flush()
                    self._last_flush = now
            except OSError as e:
                print(f"[video_inference] jsonl write falhou: {e}")

        return dets

    # ── Read-only access for SSE ────────────────────────────────────────────
    def latest(self) -> Dict:
        with self._lock:
            return {
                "t":       self._latest_t_ms,
                "segment": self._latest_seg,
                "dets":    list(self._latest_dets),
                "model":   self.model_name,
            }
