# TDL SDK — Documentação da API Python

O **TDL SDK** (Tiny Deep Learning SDK) é o framework de inferência de IA do Milk-V Duo,
desenvolvido sobre o hardware de aceleração TPU do CV181X/SG200X. A API Python expõe
todas as capacidades do SDK via o módulo `tdl`, organizado em três submódulos:

| Submódulo | Propósito |
|-----------|-----------|
| `tdl.image` | Imagens, câmera, RTSP, desenho |
| `tdl.nn` | Modelos de inferência, tracker, matcher |
| `tdl.utils` | Utilitários (tokenização BPE) |

---

## Índice

1. [Instalação e Importação](#1-instalação-e-importação)
2. [tdl.image — Imagens e Câmera](#2-tdlimage--imagens-e-câmera)
   - [ImageFormat](#imageformat)
   - [TDLDataType](#tdldatatype)
   - [Image](#image)
   - [Camera](#camera)
   - [read / write / resize / crop / crop_resize](#funções-de-imagem)
   - [align_face](#align_face)
   - [draw_bbox / draw_text / draw_detections / draw_keypoints / draw_classification / draw_segmentation / draw_instance_segmentation / draw_ocr](#funções-de-desenho)
   - [frame_to_jpeg](#frame_to_jpeg)
   - [RTSPServer](#rtspserver)
   - [RtspClient](#rtspclient)
   - [RtspClientVdec](#rtspclientvdec)
3. [tdl.nn — Redes Neurais](#3-tdlnn--redes-neurais)
   - [ModelType](#modeltype)
   - [get_model / get_model_from_dir](#get_model--get_model_from_dir)
   - [get_model_types / get_available_model_types / get_model_filename](#utilitários-de-factory)
   - [Model](#model)
   - [Formatos de saída da inferência](#formatos-de-saída-da-inferência)
   - [ObjectType](#objecttype)
   - [TrackerConfig / Tracker](#trackerconfig--tracker)
   - [Matcher](#matcher)
4. [tdl.utils](#4-tdlutils)
5. [Exemplos Completos](#5-exemplos-completos)

---

## 1. Instalação e Importação

O módulo `tdl` está disponível na imagem compilada em `/mnt/system/usr/bin/python/`.
Não requer instalação adicional no dispositivo.

```python
import tdl
from tdl import image, nn
```

---

## 2. tdl.image — Imagens e Câmera

### ImageFormat

Enumeração que define o formato de cor e layout de memória de uma imagem.

| Valor | Descrição |
|-------|-----------|
| `RGB_PACKED` | RGB intercalado (R G B R G B …) |
| `BGR_PACKED` | BGR intercalado — padrão OpenCV |
| `RGB_PLANAR` | Planos separados R…G…B… |
| `BGR_PLANAR` | Planos separados B…G…R… |
| `GRAY` | Escala de cinza, 1 canal |
| `YUV420SP_UV` | NV12 — plano Y + plano intercalado UV |
| `YUV420SP_VU` | NV21 — plano Y + plano intercalado VU |
| `YUV420P_UV` | YUV420 planar (YU12) |
| `YUV420P_VU` | YVU420 planar (YV12) |
| `YUV422SP_UV` | YUV422 semi-planar NV16 |
| `YUV422SP_VU` | YUV422 semi-planar NV61 |
| `YUV422P_UV` / `YUV422P_VU` | YUV422 planar |

> **Nota:** A câmera de hardware retorna frames em `YUV420SP_VU` (NV21).
> O servidor RTSP e as funções de desenho requerem esse formato.

---

### TDLDataType

Tipo de dado dos tensores internos.

| Valor | Descrição |
|-------|-----------|
| `UINT8` | Inteiro sem sinal 8 bits |
| `INT8` | Inteiro com sinal 8 bits |
| `UINT16` | Inteiro sem sinal 16 bits |
| `INT16` | Inteiro com sinal 16 bits |
| `UINT32` | Inteiro sem sinal 32 bits |
| `INT32` | Inteiro com sinal 32 bits |
| `FP32` | Ponto flutuante 32 bits |

---

### Image

Classe que encapsula uma imagem para uso com os modelos TDL.

#### Criação

```python
# A partir de um arquivo
img = tdl.image.read("/caminho/imagem.jpg")

# A partir de um array NumPy
import numpy as np
arr = np.zeros((480, 640, 3), dtype=np.uint8)
img = tdl.image.Image.from_numpy(arr, tdl.image.ImageFormat.BGR_PACKED)

# Versão alternativa (função no submodule)
img = tdl.image.from_numpy(arr, tdl.image.ImageFormat.BGR_PACKED)
```

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `img.get_size()` | `(width, height)` | Dimensões da imagem |
| `img.get_format()` | `ImageFormat` | Formato de cor atual |

---

### Camera

Interface de captura de câmera de hardware via VPSS (Video Processing SubSystem).
Os frames retornados ficam em memória física (ION) e são compatíveis com RTSP e
funções de desenho sem cópia adicional.

#### Construtor

```python
cam = tdl.image.Camera(
    width,                               # int — largura em pixels
    height,                              # int — altura em pixels
    format=ImageFormat.YUV420SP_VU,      # formato de saída
    vb_buffer_num=3,                     # número de buffers de vídeo
    mirror=False,                        # flip horizontal esquerda↔direita (VPSS hardware)
    flip=False,                          # flip vertical cima↔baixo (VPSS hardware)
)
```

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `cam.read(channel=0)` | `Image` | Captura um frame. **Deve ser seguido de `release()`** |
| `cam.release(channel=0)` | — | Devolve o buffer ao pool de hardware |
| `cam.close()` | — | Para a câmera e libera recursos |

> **Importante:** Chame sempre `cam.release()` após processar o frame e antes do
> próximo `cam.read()`. Não liberar trava o pool de buffers.

#### Uso como context manager

```python
with tdl.image.Camera(1280, 720) as cam:
    frame = cam.read()
    # ... processar frame ...
    cam.release()
# cam.close() é chamado automaticamente
```

#### Exemplo

```python
from tdl import image

cam = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)

for _ in range(100):
    frame = cam.read()
    # processar ou transmitir o frame
    cam.release()

cam.close()
```

---

### Funções de Imagem

#### `tdl.image.read(path)`

Carrega uma imagem de um arquivo (JPEG, PNG, BMP, etc.) e retorna um objeto `Image`.
Internamente usa OpenCV; o formato de saída é `BGR_PACKED`.

```python
img = tdl.image.read("/root/cv181x/foto.jpg")
w, h = img.get_size()   # ex: (640, 480)
```

---

#### `tdl.image.write(image, path)`

Salva uma imagem em arquivo.

```python
tdl.image.write(img, "/tmp/resultado.jpg")
```

---

#### `tdl.image.resize(src, width, height)`

Redimensiona uma imagem para as dimensões especificadas.

```python
img_pequena = tdl.image.resize(img, 320, 240)
```

---

#### `tdl.image.crop(src, roi)`

Recorta uma região retangular. `roi` é uma tupla `(x, y, width, height)`.

```python
regiao = tdl.image.crop(img, (100, 50, 200, 150))
```

---

#### `tdl.image.crop_resize(src, roi, width, height)`

Recorta e redimensiona em uma única operação.

```python
face = tdl.image.crop_resize(img, (x1, y1, x2-x1, y2-y1), 112, 112)
```

---

### align_face

#### `tdl.image.align_face(image, src_landmark_xy, dst_landmark_xy, num_points)`

Aplica uma transformação afim para alinhar um rosto detectado com base em pontos
de referência (*landmarks*). Usado antes de extrair *embeddings* faciais.

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `image` | `Image` | Imagem fonte |
| `src_landmark_xy` | `list[float]` | Coordenadas dos landmarks detectados `[x0,y0,x1,y1,...]` |
| `dst_landmark_xy` | `list[float]` | Coordenadas de destino do template `[x0,y0,x1,y1,...]` |
| `num_points` | `int` | Número de pontos (tipicamente 5 para SCRFD) |

```python
# Template de alinhamento para rosto 112×112 (5 pontos SCRFD)
DST_PTS = [
    38.2946, 51.6963,
    73.5318, 51.5014,
    56.0252, 71.7366,
    41.5493, 92.3655,
    70.7299, 92.2041,
]

detections = model_face.inference(img)
for det in detections:
    src_pts = []
    for pt in det["landmarks"]:
        src_pts.extend(pt)   # [x, y, x, y, ...]
    face_aligned = tdl.image.align_face(img, src_pts, DST_PTS, 5)
```

---

### Funções de Desenho

Todas as funções de desenho operam **diretamente** nos planos YUV do frame de
hardware (memória ION) sem conversão para BGR. Requerem um frame obtido via
`Camera.read()` (tipo `VPSSImage`).

---

#### `tdl.image.draw_bbox(frame, x1, y1, x2, y2, color=(0,255,0), thickness=2)`

Desenha um retângulo de bounding box.

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `frame` | `Image` | — | Frame de hardware |
| `x1, y1, x2, y2` | `int` | — | Coordenadas do retângulo |
| `color` | `(R, G, B)` | `(0,255,0)` | Cor em RGB |
| `thickness` | `int` | `2` | Espessura em pixels |

```python
tdl.image.draw_bbox(frame, 100, 50, 300, 250, color=(255, 80, 80), thickness=3)
```

---

#### `tdl.image.draw_text(frame, text, x, y, color=(0,255,0), scale=0.5)`

Renderiza texto no frame usando fonte HERSHEY_SIMPLEX via OpenCV (canal Y apenas).

```python
tdl.image.draw_text(frame, "Pessoa 0.92", 100, 45, color=(255, 255, 255), scale=0.5)
```

---

#### `tdl.image.draw_detections(frame, detections, score_threshold=0.0)`

Desenha automaticamente bounding boxes e labels para uma lista de detecções
retornada por `Model.inference()`. Cada classe recebe uma cor distinta da paleta
interna (12 cores). O texto do label tem contraste adaptativo (texto branco em
fundos escuros, texto preto em fundos claros).

```python
dets = model.inference(frame)
tdl.image.draw_detections(frame, dets, score_threshold=0.5)
```

---

#### `tdl.image.draw_keypoints(frame, detections, score_threshold=0.0)`

Desenha pontos-chave (*keypoints*) e linhas de esqueleto. Quando há 17 keypoints
(pose COCO-17), desenha automaticamente as 16 conexões do esqueleto corporal.

```python
dets = model_pose.inference(frame)
tdl.image.draw_keypoints(frame, dets, score_threshold=0.3)
```

---

#### `tdl.image.draw_classification(frame, result)`

Desenha o resultado de classificação ou atributos no canto superior esquerdo do
frame.

```python
result = model_cls.inference(frame)
tdl.image.draw_classification(frame, result)
```

---

#### `tdl.image.draw_segmentation(frame, result, alpha=0.5)`

Desenha overlay de segmentação semântica (saída `TOPFORMER_SEG_*`).

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `frame` | `Image` | — | Frame de hardware |
| `result` | `list[dict]` | — | Saída de `model.inference()` |
| `alpha` | `float` | `0.5` | Fator de mistura da máscara (0.0–1.0) |

```python
result = model_seg.inference(frame)
tdl.image.draw_segmentation(frame, result, alpha=0.5)
```

---

#### `tdl.image.draw_instance_segmentation(frame, result, score_threshold=0.0, alpha=0.45)`

Desenha bounding boxes e máscaras de segmentação de instâncias (saída
`YOLOV8_SEG_*`).

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `frame` | `Image` | — | Frame de hardware |
| `result` | `list[dict]` | — | Saída de `model.inference()` |
| `score_threshold` | `float` | `0.0` | Filtra detecções abaixo deste score |
| `alpha` | `float` | `0.45` | Fator de mistura da máscara |

```python
result = model_iseg.inference(frame)
tdl.image.draw_instance_segmentation(frame, result, score_threshold=0.5)
```

---

#### `tdl.image.draw_ocr(frame, result)`

Desenha o texto reconhecido por OCR (saída `RECOGNITION_LICENSE_PLATE`) na parte
inferior do frame.

```python
result = model_ocr.inference(frame)    # ex: ["ABC-1234"]
tdl.image.draw_ocr(frame, result)
```

---

### frame_to_jpeg

#### `tdl.image.frame_to_jpeg(frame, quality=80, scale=1.0)`

Converte um frame de hardware (NV21) para bytes JPEG. Útil para servir imagens
via HTTP ou gravar snapshots. A conversão usa OpenCV (software — CPU).

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `frame` | `Image` | — | Frame de câmera |
| `quality` | `int` | `80` | Qualidade JPEG (0–100) |
| `scale` | `float` | `1.0` | Fator de escala antes de codificar (ex: `0.5` = metade) |

```python
jpeg_bytes = tdl.image.frame_to_jpeg(frame, quality=75, scale=0.5)

# Servir via HTTP
from http.server import BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        self.wfile.write(jpeg_bytes)
```

---

### RTSPServer

Servidor RTSP que codifica frames de câmera em H.264 ou H.265 usando o bloco de
hardware VENC do CV181X. A codificação é 100% em hardware — a CPU não participa
na compressão de vídeo.

#### Construtor

```python
rtsp = tdl.image.RTSPServer(
    width,                  # int — largura do vídeo
    height,                 # int — altura do vídeo
    chn=0,                  # int — canal VENC (0 por padrão)
    codec="h264",           # str — "h264" ou "h265"
    session_name="live",    # str — nome da sessão (parte final da URL)
    bitrate=3072,           # int — bitrate em kbps (padrão 3072)
    gop=15,                 # int — intervalo de keyframe em frames (padrão 15)
    fps=25,                 # int — frame rate real da aplicação (padrão 25)
)
# Acesse em: rtsp://<ip-do-dispositivo>:554/live
```

> **IMPORTANTE — parâmetro `fps`:** deve corresponder ao FPS real com que a
> aplicação chama `send_frame()`. O encoder CBR usa esse valor para alocar bits
> por frame. Se `fps=25` mas a câmera opera a 12 fps, o encoder aloca apenas
> ~48% dos bits necessários por frame, resultando em vídeo com blocos ou regiões
> congeladas. Passe sempre o FPS real (ex: `fps=12` para câmera a 12 fps).

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `rtsp.send_frame(frame)` | — | Codifica e transmite um frame |
| `rtsp.get_session_name()` | `str` | Retorna o nome da sessão RTSP |

> **Restrição:** `send_frame()` requer um frame obtido diretamente de
> `Camera.read()` ou `RtspClientVdec.read()`. Frames criados a partir de
> `Image.from_numpy()` não são compatíveis (precisam estar em memória física ION).

#### Exemplo

```python
from tdl import image

rtsp = image.RTSPServer(1280, 720, codec="h264", session_name="stream",
                        bitrate=3072, gop=15, fps=15)
cam  = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)

print("Acesse: rtsp://<ip>:554/stream")

for _ in range(500):
    frame = cam.read()
    rtsp.send_frame(frame)
    cam.release()

cam.close()
del rtsp
```

---

### RtspClient

Cliente RTSP por software usando OpenCV/FFmpeg. Disponível apenas em builds com
`HAVE_OPENCV_VIDEOIO`.

#### Construtor

```python
client = tdl.image.RtspClient(
    url,                    # str — URL RTSP ou caminho de arquivo de vídeo
    width=0,                # int — redimensionar para esta largura (0 = nativo)
    height=0,               # int — redimensionar para esta altura (0 = nativo)
    timeout_ms=5000,        # int — timeout de conexão/leitura em ms
    transport="tcp",        # str — "tcp" (padrão) ou "udp"
)
```

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `client.read()` | `Image` | Decodifica o próximo frame. Lança `RuntimeError` no fim do stream |
| `client.release()` | — | No-op (compatibilidade de API com `Camera`) |
| `client.close()` | — | Fecha o stream e libera o decoder |
| `client.is_opened()` | `bool` | `True` se o stream está aberto |

---

### RtspClientVdec

Cliente RTSP com decodificação H.264/H.265 em hardware (live555 + VDEC).
Frames decodificados ficam em memória VB (NV12) compatíveis com `model.inference()`
e `RTSPServer.send_frame()` sem cópia de dados.

O CV181X suporta **1 canal VDEC**. Se o hardware não estiver disponível, use
`RtspClient` como fallback.

#### Construtor

```python
client = tdl.image.RtspClientVdec(
    url,                    # str — URL RTSP (rtsp://...)
    width=0,                # int — resolução máxima de decode (0 = nativo)
    height=0,
    timeout_ms=5000,        # int — timeout por frame em ms
    transport="tcp",        # str — "tcp" (padrão) ou "udp"
)
```

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `client.read()` | `Image` | Obtém o próximo frame via VDEC hardware (NV12). **Deve ser seguido de `release()`** |
| `client.release()` | — | Devolve o frame atual ao pool VDEC. Obrigatório antes do próximo `read()` |
| `client.pin_for_inference()` | — | Move o frame atual para o slot de inferência, liberando `read()` para buscar o próximo frame |
| `client.release_inference()` | — | Libera o slot de inferência de volta ao pool VDEC após a thread de inferência terminar |
| `client.close()` | — | Para o stream e libera todos os recursos |
| `client.is_opened()` | `bool` | `True` se o stream está aberto e o VDEC está rodando |

#### Exemplo — com fallback automático

```python
from tdl import image
import sys

def open_rtsp(url, width=0, height=0):
    try:
        client = image.RtspClientVdec(url, width=width, height=height)
        if client.is_opened():
            print("Backend: vdec (hardware H264 decode)")
            return client
    except Exception as e:
        print(f"VDEC indisponível ({e}), usando OpenCV")
    if not hasattr(image, "RtspClient"):
        print("Erro: backend opencv não disponível nesta build")
        sys.exit(1)
    print("Backend: opencv (decode por software)")
    return image.RtspClient(url, width=width, height=height)

client = open_rtsp("rtsp://192.168.1.10:554/live", width=960, height=576)

while True:
    frame = client.read()
    # processar frame...
    client.release()
```

---

## 3. tdl.nn — Redes Neurais

### ModelType

Enumeração com todos os tipos de modelos suportados pelo SDK. O valor é passado
para `get_model()` ou `get_model_from_dir()`.

#### Detecção de Objetos (YOLO)

| ModelType | Classes | Descrição |
|-----------|---------|-----------|
| `YOLOV8_DET_COCO80` | 80 (COCO) | YOLOv8 — 80 classes padrão COCO |
| `YOLOV8N_DET_PERSON_VEHICLE` | 7 | Pessoas e veículos |
| `YOLOV8N_DET_HAND_FACE_PERSON` | 3 | Mão, rosto, pessoa |
| `YOLOV8N_DET_HEAD_PERSON` | 2 | Cabeça, pessoa |
| `YOLOV8N_DET_FIRE_SMOKE` | 2 | Fogo, fumaça |
| `YOLOV8N_DET_HAND` | 1 | Mão |
| `YOLOV8N_DET_FIRE` | 1 | Fogo |
| `YOLOV8N_DET_LICENSE_PLATE` | 1 | Placa veicular |
| `YOLOV8N_DET_PET_PERSON` | 3 | Gato, cachorro, pessoa |
| `YOLOV8N_DET_HEAD_HARDHAT` | 2 | Cabeça, capacete de segurança |
| `YOLOV8N_DET_TRAFFIC_LIGHT` | 5 | Semáforos (red/yellow/green/off/wait) |
| `YOLOV8N_DET_BICYCLE_MOTOR_EBICYCLE` | 3 | Bicicleta, moto, e-bike |
| `YOLOV8N_DET_HEAD_SHOULDER` | 1 | Cabeça+ombros |
| `YOLOV8N_DET_MONITOR_PERSON` | 1 | Pessoa (monitoramento) |
| `YOLOV8N_DET_FACE_HEAD_PERSON_PET` | 4 | Rosto, cabeça, pessoa, pet |
| `YOLOV11N_DET_COCO80` | 80 (COCO) | YOLO11 — 80 classes COCO |
| `YOLOV11N_DET_PERSON_VEHICLE` | 7 | Pessoas e veículos |
| `YOLOV11N_DET_HAND_FACE_PERSON` | 3 | Mão, rosto, pessoa |
| `YOLOV11N_DET_HEAD_PERSON` | 2 | Cabeça, pessoa |
| `YOLOV11N_DET_FIRE_SMOKE` | 2 | Fogo, fumaça |
| `YOLOV11N_DET_MONITOR_PERSON` | 1 | Pessoa (monitoramento) |
| `YOLOV11N_DET_BICYCLE_MOTOR_EBICYCLE` | 3 | Bicicleta, moto, e-bike |
| `YOLOV11` | — | YOLO11 genérico (classes via config) |
| `YOLOV26_DET_COCO80` | 80 (COCO) | YOLO26 — 80 classes COCO |
| `YOLOV26_DET_PERSON_VEHICLE` | 7 | Pessoas e veículos |
| `YOLOV26_DET_HAND_FACE_PERSON` | 3 | Mão, rosto, pessoa |
| `YOLOV26_DET_HEAD_PERSON` | 2 | Cabeça, pessoa |
| `YOLOV26_DET_FIRE_SMOKE` | 2 | Fogo, fumaça |
| `YOLOV26` | — | YOLO26 genérico |
| `YOLOV10_DET_COCO80` | 80 (COCO) | YOLOv10 |
| `YOLOV7_DET_COCO80` | 80 (COCO) | YOLOv7 |
| `YOLOV6_DET_COCO80` | 80 (COCO) | YOLOv6 |
| `YOLOV5_DET_COCO80` | 80 (COCO) | YOLOv5 |
| `PPYOLOE_DET_COCO80` | 80 (COCO) | PP-YOLOE |
| `YOLOX_DET_COCO80` | 80 (COCO) | YOLOX |
| `MBV2_DET_PERSON` | 1 | MobileDetV2 — pessoa |

#### Detecção de Rostos

| ModelType | Saída | Descrição |
|-----------|-------|-----------|
| `SCRFD_DET_FACE` | bbox + 5 landmarks | SCRFD — detector de rosto com keypoints |

#### Detecção de Keypoints / Pose

| ModelType | Pontos | Descrição |
|-----------|--------|-----------|
| `KEYPOINT_YOLOV8POSE_PERSON17` | 17 (COCO) | Estimação de pose com bbox |
| `KEYPOINT_SIMCC_PERSON17` | 17 (COCO) | SimCC pose (entrada: crop de pessoa) |
| `KEYPOINT_HAND` | 21 | 21 pontos de mão |
| `KEYPOINT_LICENSE_PLATE` | 4 | Quatro cantos de placa |
| `KEYPOINT_FACE_V2` | N | Landmarks faciais V2 |

#### Classificação

| ModelType | Descrição |
|-----------|-----------|
| `CLS_HAND_GESTURE` | Classificação de gestos de mão |
| `CLS_KEYPOINT_HAND_GESTURE` | Gesto a partir de keypoints de mão |
| `CLS_RGBLIVENESS` | Detecção de vivacidade (anti-spoofing) |
| `CLS_ATTRIBUTE_GENDER_AGE_GLASS` | Gênero, idade, óculos |
| `CLS_ATTRIBUTE_GENDER_AGE_GLASS_MASK` | + máscara facial |
| `CLS_ATTRIBUTE_GENDER_AGE_GLASS_EMOTION` | + emoção |
| `CLS_ISP_SCENE` | Classificação de cena ISP |
| `CLS_YOLOV8` | Classificação genérica YOLOv8 |
| `CLS_IMG` | Classificação genérica de imagem |
| `CLS_SOUND_BABAY_CRY` | Classificação de áudio — choro de bebê |
| `CLS_SOUND_COMMAND` | Reconhecimento de comando de voz |
| `CLS_SOUND_COMMAND_NIHAOSHIYUN` | Comando "Ni Hao Shi Yun" |
| `CLS_SOUND_COMMAND_NIHAOSUANNENG` | Comando "Ni Hao Suan Neng" |
| `CLS_SOUND_COMMAND_XIAOAIXIAOAI` | Comando "Xiao Ai Xiao Ai" |

#### Segmentação

| ModelType | Descrição |
|-----------|-----------|
| `YOLOV8_SEG_COCO80` | Segmentação de instâncias — 80 classes COCO |
| `YOLOV8_SEG` | Segmentação de instâncias genérica |
| `TOPFORMER_SEG_PERSON_FACE_VEHICLE` | Segmentação semântica: pessoa, rosto, veículo |
| `TOPFORMER_SEG_MOTION` | Segmentação de movimento |

#### Extração de Features

| ModelType | Dimensão | Descrição |
|-----------|----------|-----------|
| `FEATURE_CVIFACE` | 256 | Embedding facial (Sophgo) |
| `FEATURE_BMFACE_R34` | 512 | Embedding facial ResNet-34 |
| `FEATURE_BMFACE_R50` | 512 | Embedding facial ResNet-50 |
| `FEATURE_IMG` | — | Embedding de imagem genérico |
| `FEATURE_CLIP_IMG` | — | CLIP — encoder de imagem |
| `FEATURE_CLIP_TEXT` | — | CLIP — encoder de texto |
| `FEATURE_MOBILECLIP2_IMG` | — | MobileCLIP2 — encoder de imagem |
| `FEATURE_MOBILECLIP2_TEXT` | — | MobileCLIP2 — encoder de texto |

#### Outros

| ModelType | Descrição |
|-----------|-----------|
| `LSTR_DET_LANE` | Detecção de faixas de estrada (LSTR) |
| `RECOGNITION_LICENSE_PLATE` | OCR de placa veicular |
| `TRACKING_FEARTRACK` | Tracker de objeto único (SOT) |
| `RECOGNITION_SPEECH_ZIPFORMER_ENCODER` | ASR — encoder Zipformer |
| `RECOGNITION_SPEECH_ZIPFORMER_DECODER` | ASR — decoder Zipformer |
| `RECOGNITION_SPEECH_ZIPFORMER_JOINER` | ASR — joiner Zipformer |

---

### get_model / get_model_from_dir

#### `tdl.nn.get_model(model_type, model_path, model_config={}, device_id=0)`

Carrega um modelo a partir de um arquivo `.cvimodel` específico.

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model_type` | `ModelType` | — | Tipo do modelo |
| `model_path` | `str` | — | Caminho completo para o `.cvimodel` |
| `model_config` | `dict` | `{}` | Configuração de pré-processamento (opcional) |
| `device_id` | `int` | `0` | ID do dispositivo TPU |

```python
from tdl import nn

model = nn.get_model(
    nn.ModelType.SCRFD_DET_FACE,
    "/root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel"
)
```

**`model_config`** permite sobrescrever os parâmetros de pré-processamento:

```python
model = nn.get_model(
    nn.ModelType.YOLOV8,
    "/root/cv181x/meu_modelo.cvimodel",
    model_config={
        "mean": (123.675, 116.28, 103.53),
        "scale": (58.395, 57.12, 57.375),
        "rgb_order": "rgb",
    }
)
```

---

#### `tdl.nn.get_model_from_dir(model_type, model_dir="", device_id=0)`

Carrega um modelo a partir de um diretório, usando o arquivo
`configs/model/model_factory.json` para resolver o nome do arquivo `.cvimodel`
a partir do `ModelType`.

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model_type` | `ModelType` | — | Tipo do modelo |
| `model_dir` | `str` | `""` | Diretório onde estão os `.cvimodel` |
| `device_id` | `int` | `0` | ID do dispositivo TPU |

```python
model = nn.get_model_from_dir(
    nn.ModelType.YOLOV8N_DET_PERSON_VEHICLE,
    "/root/cv181x"
)
```

---

### Utilitários de Factory

Funções auxiliares que consultam o `model_factory.json` sem precisar carregar o
modelo.

#### `tdl.nn.get_model_types(model_type_name)` → `list[str]`

Retorna a lista de nomes de classes definida para um `ModelType` no
`model_factory.json`. Retorna lista vazia se o tipo não tiver classes (ex: modelos
genéricos como `YOLOV26`).

```python
classes = nn.get_model_types("YOLOV8_DET_COCO80")
# ['person', 'bicycle', 'car', ...]   # 80 classes COCO
```

#### `tdl.nn.get_available_model_types()` → `list[str]`

Retorna todos os nomes de `ModelType` presentes no `model_factory.json`.

```python
types = nn.get_available_model_types()
for t in types:
    print(t)
```

#### `tdl.nn.get_model_filename(model_type_name)` → `str`

Retorna o `file_name` base do modelo como definido no `model_factory.json`
(sem o sufixo `_<platform>.cvimodel`). Útil para construir o caminho completo
dinamicamente.

```python
base = nn.get_model_filename("SCRFD_DET_FACE")
# ex: "scrfd_det_face_432_768_INT8"
path = f"/root/cv181x/{base}_cv181x.cvimodel"
```

---

### Model

Classe de inferência retornada por `get_model()` / `get_model_from_dir()`.

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `model.inference(image)` | `list` | Inferência sobre `Image` |
| `model.inference(array)` | `list` | Inferência sobre array NumPy `uint8` |
| `model.inference(image, params)` | `list` | Inferência com parâmetros extras |
| `model.set_threshold(float)` | — | Define limiar de confiança |
| `model.get_threshold()` | `float` | Retorna limiar atual |
| `model.set_soft_nms(enable, sigma=0.5)` | — | Ativa Soft NMS Gaussiano. `sigma`: taxa de decaimento (padrão 0.5) |
| `model.get_soft_nms()` | `bool` | Retorna se Soft NMS está ativo |
| `model.get_input_names()` | `list[str]` | Nomes das camadas de entrada |
| `model.get_output_names()` | `list[str]` | Nomes das camadas de saída |
| `model.get_preprocess_parameters()` | `dict` | Retorna `{mean, scale}` do pré-processamento |
| `model.close()` | — | Libera o modelo e o grupo VPSS imediatamente |

```python
model.set_threshold(0.6)
print(model.get_threshold())           # 0.6
print(model.get_input_names())         # ['images']
print(model.get_preprocess_parameters())
# {'mean': (0.0, 0.0, 0.0), 'scale': (0.0039, 0.0039, 0.0039)}
```

---

### Formatos de Saída da Inferência

O retorno de `model.inference()` depende do tipo de modelo:

#### Detecção de Objetos

Retorna `list[dict]` com uma entrada por objeto detectado.

```python
[
    {
        "class_id":   0,          # int — índice da classe
        "class_name": "person",   # str — nome da classe
        "x1": 120.5,              # float — coordenada esquerda
        "y1":  80.3,              # float — coordenada superior
        "x2": 340.1,              # float — coordenada direita
        "y2": 510.7,              # float — coordenada inferior
        "score": 0.92,            # float — confiança [0, 1]
    },
    ...
]
```

> **Modelos COCO80** (`YOLOV8_DET_COCO80`, `YOLOV11N_DET_COCO80`, `YOLOV26_DET_COCO80`):
> o campo `class_name` retorna o nome padrão COCO (ex: `"person"`, `"car"`, `"dog"`).
> Outros modelos retornam o nome mapeado na configuração do factory.

---

#### Detecção de Rostos com Landmarks (SCRFD)

```python
[
    {
        "class_id":   0,
        "class_name": "face",
        "x1": 100.0, "y1": 80.0, "x2": 250.0, "y2": 280.0,
        "score": 0.97,
        "landmarks": [
            [150.2, 140.5],   # olho esquerdo
            [210.1, 138.9],   # olho direito
            [180.0, 170.3],   # nariz
            [155.0, 200.8],   # canto esquerdo da boca
            [205.0, 199.2],   # canto direito da boca
        ],
        "landmarks_score": 0.99,
    },
    ...
]
```

---

#### Pose / Keypoints (YOLOV8POSE, SIMCC)

```python
[
    {
        "class_id": 0, "class_name": "person",
        "x1": 50.0, "y1": 30.0, "x2": 300.0, "y2": 580.0,
        "score": 0.88,
        "landmarks": [
            [180.0, 60.0],    # 0: nariz
            [165.0, 55.0],    # 1: olho esquerdo
            # ... 17 pontos COCO
        ],
        "landmarks_score": [0.95, 0.92, ...],
    },
    ...
]
```

---

#### Classificação

```python
[
    {
        "class_id": 3,
        "score":    0.87,
    }
]
```

---

#### Atributos Faciais

```python
[
    {
        "gender_score":       0.91,   # > 0.5 = masculino
        "is_male":            True,
        "age_score":          0.28,
        "age":                28,     # int(age_score * 100)
        "glass_score":        0.15,
        "is_wearing_glasses": False,
        "mask_score":         0.04,
        "is_wearing_mask":    False,
    }
]
```

---

#### Embedding / Feature

Retorna `numpy.ndarray` com o vetor de características.

```python
embedding = model.inference(face_img)  # np.ndarray shape (256,) dtype float32
```

---

#### Segmentação de Instâncias (YOLOv8-seg)

```python
[
    {
        "mask_width":  160,
        "mask_height": 160,
        "bboxes_seg": [
            {
                "class_id": 0, "class_name": "person",
                "x1": 50.0, "y1": 30.0, "x2": 300.0, "y2": 580.0,
                "score": 0.88,
                "mask": [0, 0, 1, 1, ...],  # 160×160 ints (0 ou 1)
            },
            ...
        ]
    }
]
```

---

#### Segmentação Semântica (Topformer)

```python
[
    {
        "output_width":  80,
        "output_height": 45,
        "class_id":   [0, 0, 1, 2, ...],  # índice de classe por pixel
        "class_conf": [0.9, 0.8, ...],    # confiança por pixel
    }
]
```

---

#### OCR — Reconhecimento de Placa

```python
["ABC-1234"]   # list[str]
```

---

### ObjectType

Enumeração de tipos de objetos para o tracker.

| Valor | Descrição |
|-------|-----------|
| `UNDEFINED` | Indefinido |
| `PERSON` | Pessoa |
| `FACE` | Rosto |
| `HAND` | Mão |
| `HEAD` | Cabeça |
| `HEAD_SHOULDER` | Cabeça + ombros |
| `HARD_HAT` | Capacete de segurança |
| `FACE_MASK` | Máscara facial |
| `CAR` | Carro |
| `BUS` | Ônibus |
| `TRUCK` | Caminhão |
| `MOTORBIKE` | Motocicleta |
| `BICYCLE` | Bicicleta |
| `LICENSE_PLATE` | Placa veicular |
| `FIRE` | Fogo |
| `SMOKE` | Fumaça |

---

### TrackerConfig / Tracker

Tracker multi-objeto baseado em SORT (Simple Online Realtime Tracking).

#### TrackerConfig

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `max_unmatched_times` | `int` | Frames sem correspondência antes de remover a trilha |
| `track_confirmed_frames` | `int` | Frames necessários para confirmar uma nova trilha |
| `track_init_score_thresh` | `float` | Score mínimo para iniciar uma nova trilha |
| `high_score_thresh` | `float` | Threshold para associação de alta confiança |
| `high_score_iou_dist_thresh` | `float` | Limiar IoU para associação de alta confiança |
| `low_score_iou_dist_thresh` | `float` | Limiar IoU para associação de baixa confiança |

#### Tracker

```python
tracker = tdl.nn.Tracker(tdl.nn.TrackerType.MOT_SORT)
tracker.set_img_size(1280, 720)

# Ajustar configuração
cfg = tracker.get_track_config()
cfg.max_unmatched_times  = 10
cfg.track_confirmed_frames = 2
tracker.set_track_config(cfg)
```

#### `tracker.track(boxes, frame_id)` → `list[dict]`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `track_id` | `int` | ID único e persistente da trilha |
| `status` | `int` | `0`=NEW, `1`=TRACKED, `2`=LOST, `3`=REMOVED |
| `obj_idx` | `int` | Índice da detecção associada |
| `box_info` | `dict` | `{x1, y1, x2, y2, class_id, score}` |
| `velocity_x` | `float` | Velocidade horizontal (pixels/frame) |
| `velocity_y` | `float` | Velocidade vertical (pixels/frame) |

```python
from tdl import nn, image

model   = nn.get_model(nn.ModelType.YOLOV8N_DET_PERSON_VEHICLE, "/root/cv181x/model.cvimodel")
tracker = nn.Tracker(nn.TrackerType.MOT_SORT)
tracker.set_img_size(1280, 720)

cam = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)
frame_id = 0

while True:
    frame = cam.read()
    dets  = model.inference(frame)
    tracks = tracker.track(dets, frame_id)

    for t in tracks:
        if t["status"] == 1:   # TRACKED
            b = t["box_info"]
            label = f"ID:{t['track_id']}"
            image.draw_bbox(frame, int(b["x1"]), int(b["y1"]),
                                   int(b["x2"]), int(b["y2"]))
            image.draw_text(frame, label, int(b["x1"]), int(b["y1"]) - 5)

    cam.release()
    frame_id += 1
```

---

### Matcher

Galeria de features para busca por similaridade (re-identificação de pessoas,
reconhecimento facial, etc.).

#### Construtor

```python
matcher = tdl.nn.Matcher("cosine")      # similaridade cosseno
# ou
matcher = tdl.nn.Matcher("euclidean")   # distância euclidiana
```

#### Métodos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `matcher.load_gallery(features)` | — | Carrega galeria. `features`: `list[np.ndarray]` |
| `matcher.query(features, topk=1)` | `(indices, scores)` | Busca os `topk` mais similares |
| `matcher.update_gallery(features, col)` | — | Atualiza a coluna `col` da galeria |
| `matcher.get_gallery_size()` | `int` | Número de entradas na galeria |
| `matcher.get_feature_dim()` | `int` | Dimensão dos vetores de feature |

```python
import numpy as np
from tdl import nn

feat_model = nn.get_model(nn.ModelType.FEATURE_CVIFACE, "/root/cv181x/feature_cviface.cvimodel")

# Construir galeria
gallery = [feat_model.inference(face_img) for face_img in galeria_rostos]

matcher = nn.Matcher("cosine")
matcher.load_gallery(gallery)

# Consulta
query_feat = feat_model.inference(rosto_desconhecido)
indices, scores = matcher.query([query_feat], topk=3)

for idx, score in zip(indices[0], scores[0]):
    print(f"  Match: índice={idx}  similaridade={score:.3f}")
```

---

## 4. tdl.utils

### BytePairEncoder

Tokenizador BPE (Byte Pair Encoding) para modelos de linguagem.

```python
bpe = tdl.utils.BytePairEncoder("encoder.json", "vocab.bpe")
tokens = bpe.tokenizer_bpe("texto.txt")   # list[list[int]]
```

---

## 5. Exemplos Completos

### Detecção de Objetos em Imagem Estática

```python
#!/usr/bin/env python3
"""
Detecção de objetos em uma imagem estática.
Uso: python3 deteccao.py /root/cv181x/yolov8n_det_coco80_640_640_INT8_cv181x.cvimodel imagem.jpg
"""
import sys
from tdl import nn, image

model_path = sys.argv[1]
image_path = sys.argv[2]

# Carregar modelo
model = nn.get_model(nn.ModelType.YOLOV8_DET_COCO80, model_path)
model.set_threshold(0.5)

# Ler imagem e inferir
img  = image.read(image_path)
dets = model.inference(img)

print(f"{len(dets)} objeto(s) detectado(s):")
for d in dets:
    print(f"  {d['class_name']:20s}  score={d['score']:.2f}"
          f"  bbox=({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f})")
```

---

### Servidor RTSP com Detecção em Tempo Real

```python
#!/usr/bin/env python3
"""
Stream RTSP com overlay de detecção de rostos.
Acesse: rtsp://<ip>:554/live
"""
import threading
import time
from tdl import nn, image

MODEL = "/root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel"
CAM_FPS = 12   # FPS real da câmera — deve coincidir com o parâmetro fps do RTSPServer

model = nn.get_model(nn.ModelType.SCRFD_DET_FACE, MODEL)
model.set_threshold(0.5)

# fps=CAM_FPS é essencial: controla a alocação de bits do encoder CBR.
# Valor errado causa blocos ou regiões congeladas no vídeo.
rtsp = image.RTSPServer(1280, 720, codec="h264", session_name="live",
                        bitrate=3072, gop=15, fps=CAM_FPS)
cam  = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)

# Fila de inferência: cada item é (frame, release_fn)
# A thread de inferência chama release_fn() após terminar o uso do frame,
# evitando que o buffer VB seja liberado enquanto o VPSS ainda o lê.
_last_dets = []
_det_lock  = threading.Lock()
_queue     = []     # lista de (frame, release_fn)
_q_lock    = threading.Lock()
_running   = True

def inference_worker():
    while _running:
        item = None
        with _q_lock:
            if _queue:
                item = _queue.pop(0)
        if item is None:
            time.sleep(0.001)
            continue
        frame, release_fn = item
        dets = model.inference(frame)
        release_fn()   # libera o VB block após VPSS + NPU terminarem
        with _det_lock:
            _last_dets[:] = dets

t = threading.Thread(target=inference_worker, daemon=True)
t.start()

print("Streaming em rtsp://<ip>:554/live — Ctrl+C para parar")

try:
    frame_id = 0
    while True:
        frame = cam.read()

        # Enviar para inferência (1 em cada 2 frames).
        # A thread de inferência é responsável pelo cam.release() deste frame.
        sent_to_infer = False
        if frame_id % 2 == 0:
            with _q_lock:
                # Descartar frame anterior não consumido
                while _queue:
                    _, old_release = _queue.pop(0)
                    old_release()
                _queue.append((frame, cam.release))
                sent_to_infer = True

        # Desenhar detecções atuais
        with _det_lock:
            dets = list(_last_dets)
        if dets:
            image.draw_detections(frame, dets, score_threshold=0.5)

        rtsp.send_frame(frame)
        if not sent_to_infer:
            cam.release()
        frame_id += 1

except KeyboardInterrupt:
    pass
finally:
    _running = False
    t.join(2)
    with _q_lock:
        while _queue:
            _, old_release = _queue.pop(0)
            old_release()
    cam.close()
    del rtsp
```

---

### Detecção de Rostos + Extração de Features + Matching

```python
#!/usr/bin/env python3
"""
Pipeline: detecção de rosto → alinhamento → embedding → busca na galeria.
"""
import numpy as np
from tdl import nn, image

FACE_MODEL    = "/root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel"
FEATURE_MODEL = "/root/cv181x/feature_cviface_112_112_INT8_cv181x.cvimodel"

detector = nn.get_model(nn.ModelType.SCRFD_DET_FACE,  FACE_MODEL)
extractor = nn.get_model(nn.ModelType.FEATURE_CVIFACE, FEATURE_MODEL)
detector.set_threshold(0.5)

# Template SCRFD 5 pontos → rosto 112×112
DST_PTS = [
    38.2946, 51.6963, 73.5318, 51.5014, 56.0252,
    71.7366, 41.5493, 92.3655, 70.7299, 92.2041,
]

def get_face_embedding(img_path):
    img  = image.read(img_path)
    dets = detector.inference(img)
    if not dets:
        return None
    det = max(dets, key=lambda d: d["score"])
    src_pts = [c for pt in det["landmarks"] for c in pt]
    face = image.align_face(img, src_pts, DST_PTS, 5)
    return extractor.inference(face)   # np.ndarray (256,)

# Construir galeria
nomes   = ["Alice", "Bob", "Carlos"]
galeria = [get_face_embedding(f"/fotos/{n.lower()}.jpg") for n in nomes]

matcher = nn.Matcher("cosine")
matcher.load_gallery(galeria)

# Reconhecer rosto desconhecido
query = get_face_embedding("/fotos/desconhecido.jpg")
if query is not None:
    indices, scores = matcher.query([query], topk=1)
    idx, score = indices[0][0], scores[0][0]
    if score > 0.6:
        print(f"Reconhecido: {nomes[idx]} (similaridade={score:.3f})")
    else:
        print(f"Desconhecido (melhor match: {nomes[idx]}, score={score:.3f})")
```

---

### Tracking Multi-Objeto com Visualização

```python
#!/usr/bin/env python3
"""
Rastreamento de pessoas em tempo real via RTSP.
"""
from tdl import nn, image

MODEL = "/root/cv181x/yolov8n_det_person_vehicle_384_640_INT8_cv181x.cvimodel"

model   = nn.get_model(nn.ModelType.YOLOV8N_DET_PERSON_VEHICLE, MODEL)
tracker = nn.Tracker(nn.TrackerType.MOT_SORT)
tracker.set_img_size(1280, 720)

cfg = tracker.get_track_config()
cfg.max_unmatched_times  = 15
cfg.track_confirmed_frames = 3
tracker.set_track_config(cfg)

rtsp = image.RTSPServer(1280, 720, session_name="tracking")
cam  = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)

print("Stream: rtsp://<ip>:554/tracking")

frame_id = 0
try:
    while True:
        frame = cam.read()
        dets  = model.inference(frame)
        tracks = tracker.track(dets, frame_id)

        for t in tracks:
            if t["status"] not in (0, 1):   # NEW ou TRACKED
                continue
            b = t["box_info"]
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            tid = t["track_id"]
            # Cor baseada no ID (cicla entre 12 cores da paleta interna)
            image.draw_bbox(frame, x1, y1, x2, y2, thickness=2)
            image.draw_text(frame, f"#{tid}", x1, y1 - 5, scale=0.5)

        rtsp.send_frame(frame)
        cam.release()
        frame_id += 1

except KeyboardInterrupt:
    pass
finally:
    cam.close()
    del rtsp
```

---

### Segmentação de Instâncias

```python
#!/usr/bin/env python3
import numpy as np
from tdl import nn, image

MODEL = "/root/cv181x/yolov8n_seg_coco80_640_640_INT8_cv181x.cvimodel"

model = nn.get_model(nn.ModelType.YOLOV8_SEG_COCO80, MODEL)
model.set_threshold(0.4)

img = image.read("/tmp/foto.jpg")
w, h = img.get_size()

results = model.inference(img)
if results:
    seg = results[0]
    mw, mh = seg["mask_width"], seg["mask_height"]
    print(f"Máscara: {mw}×{mh}")
    for obj in seg["bboxes_seg"]:
        mask = np.array(obj["mask"], dtype=np.uint8).reshape(mh, mw)
        print(f"  {obj['class_name']:15s} score={obj['score']:.2f}"
              f"  pixels_segmentados={mask.sum()}")
```

---

### Servidor Web com JPEG (HTTP snapshot)

```python
#!/usr/bin/env python3
"""
Servidor HTTP que serve snapshots da câmera com overlay de detecção.
Acesse: http://<ip>:8080/
"""
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from tdl import nn, image

MODEL   = "/root/cv181x/scrfd_det_face_432_768_INT8_cv181x.cvimodel"
model   = nn.get_model(nn.ModelType.SCRFD_DET_FACE, MODEL)
cam     = image.Camera(1280, 720, image.ImageFormat.YUV420SP_VU)
_lock   = threading.Lock()
_jpeg   = b""

def capture_loop():
    global _jpeg
    while True:
        frame = cam.read()
        dets  = model.inference(frame)
        if dets:
            image.draw_detections(frame, dets, score_threshold=0.5)
        jpg = image.frame_to_jpeg(frame, quality=75, scale=0.5)
        with _lock:
            _jpeg = bytes(jpg)
        cam.release()

threading.Thread(target=capture_loop, daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _lock:
            data = _jpeg
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args): pass

print("Servidor em http://<ip>:8080/")
HTTPServer(("", 8080), Handler).serve_forever()
```
