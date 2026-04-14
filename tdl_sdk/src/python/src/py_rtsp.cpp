#include "py_rtsp.hpp"
#include "image/base_image.hpp"
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"
#include "opencv2/opencv.hpp"
#include <cvi_sys.h>
#include <algorithm>
#include <cstring>

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
#include "encoder/image_encoder/image_encoder.hpp"
// JPEG uses VENC channel 1 (channel 0 is reserved for H264/H265 streaming).
static ImageEncoder& hwJpegEncoder() {
  static ImageEncoder enc(1);
  return enc;
}
#endif

namespace pytdl {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static VPSSImage* requireVPSS(const PyImage& image, const char* fn) {
  VPSSImage* vpss = dynamic_cast<VPSSImage*>(image.getImage().get());
  if (!vpss)
    throw std::runtime_error(
        std::string(fn) + ": requires a hardware camera frame (VPSSImage). "
        "Use image.Camera.read() to get one.");
  return vpss;
}

// ─── YUV color ───────────────────────────────────────────────────────────────

struct YUVColor {
  uint8_t Y = 128, U = 128, V = 128;

  // From RGB (full-range BT.601)
  static YUVColor fromRGB(int r, int g, int b) {
    auto clp = [](int v) -> uint8_t { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; };
    YUVColor c;
    c.Y = clp(( 77*r + 150*g + 29*b) >> 8);
    c.U = clp(((-43*r -  85*g + 128*b) >> 8) + 128);
    c.V = clp(((128*r - 107*g -  21*b) >> 8) + 128);
    return c;
  }
  // From BGR (OpenCV Scalar order)
  static YUVColor fromBGR(int b, int g, int r) { return fromRGB(r, g, b); }
};

static YUVColor parseColorYUV(const py::tuple& color) {
  if (color.size() != 3)
    throw std::runtime_error("color must be a (R, G, B) tuple");
  return YUVColor::fromRGB(color[0].cast<int>(), color[1].cast<int>(), color[2].cast<int>());
}

// Detection color palette (RGB).
// Chosen for maximum mutual distinctiveness and good visibility on natural scenes.
// Dark colors (Y<128) get white text; light colors (Y>=128) get black text.
static YUVColor detectionColor(int cls_id) {
  static const uint8_t kRGB[][3] = {
    { 50, 205,  50},  // 0  lime green        Y≈165  → black text
    {255,  80,  80},  // 1  salmon red         Y≈112  → white text
    { 30, 144, 255},  // 2  dodger blue        Y≈120  → white text
    {255, 200,   0},  // 3  amber yellow       Y≈190  → black text
    {220,  80, 220},  // 4  orchid magenta     Y≈110  → white text
    {  0, 210, 210},  // 5  cyan               Y≈176  → black text
    {255, 140,   0},  // 6  dark orange        Y≈150  → black text
    {180, 255, 100},  // 7  yellow-green       Y≈220  → black text
    {255, 105, 180},  // 8  hot pink           Y≈140  → black text
    { 80, 200, 120},  // 9  emerald green      Y≈163  → black text
    {255, 165,   0},  // 10 orange             Y≈163  → black text
    {100, 100, 255},  // 11 periwinkle blue    Y≈107  → white text
  };
  static constexpr int N = sizeof(kRGB) / sizeof(kRGB[0]);
  const auto& c = kRGB[cls_id % N];
  return YUVColor::fromRGB(c[0], c[1], c[2]);
}

// ─── Frame plane mapping ─────────────────────────────────────────────────────
//
// We always map planes fresh from u64PhyAddr[] with CVI_SYS_MmapCache.
// This bypasses the stale-VA bug in VPSSImage::restoreVirtualAddress()
// (which incorrectly assumes the two planes are contiguous in virtual space).
// MmapCache gives cache-speed writes (~500 MB/s vs ~5 MB/s uncached).
// We only need IonFlushCache at the end (not Invalidate, since we only write).

struct PlaneMapping {
  uint8_t* y_va   = nullptr;
  uint8_t* uv_va  = nullptr;
  uint64_t y_pa   = 0;
  uint64_t uv_pa  = 0;
  uint32_t y_len  = 0;
  uint32_t uv_len = 0;

  ~PlaneMapping() { unmap(); }
  void unmap() {
    if (y_va)  { CVI_SYS_Munmap(y_va,  y_len);  y_va  = nullptr; }
    if (uv_va) { CVI_SYS_Munmap(uv_va, uv_len); uv_va = nullptr; }
  }
};

static PlaneMapping mapPlanes(VPSSImage* vpss) {
  VIDEO_FRAME_INFO_S* fi = vpss->getFrame();
  int h = (int)vpss->getHeight();
  auto st = vpss->getStrides();

  PlaneMapping m;
  m.y_pa  = fi->stVFrame.u64PhyAddr[0];
  m.uv_pa = fi->stVFrame.u64PhyAddr[1];
  m.y_len  = fi->stVFrame.u32Length[0] ? fi->stVFrame.u32Length[0] : (uint32_t)h        * st[0];
  m.uv_len = fi->stVFrame.u32Length[1] ? fi->stVFrame.u32Length[1] : (uint32_t)(h / 2)  * st[1];

  if (!m.y_pa || !m.uv_pa)
    throw std::runtime_error("draw: null physical address — call cam.read() first");

  m.y_va  = static_cast<uint8_t*>(CVI_SYS_MmapCache(m.y_pa,  m.y_len));
  m.uv_va = static_cast<uint8_t*>(CVI_SYS_MmapCache(m.uv_pa, m.uv_len));
  if (!m.y_va || !m.uv_va) { m.unmap(); throw std::runtime_error("draw: MmapCache failed"); }
  return m;
}

// ─── Direct YUV draw primitives ──────────────────────────────────────────────
//
// Draw directly onto Y and UV planes — no cvtColor, no intermediate BGR mat.
// Per-pixel UV has 2×2 chroma subsampling; adjacent pixels share UV pairs,
// which causes slight color bleeding at sub-2px boundaries but is imperceptible
// for bounding-box overlays and text labels.

// Set one pixel in Y and UV planes.
static inline void yuvSetPixel(uint8_t* yp, uint8_t* uvp, int ys, int us, bool nv21,
                                int x, int y, const YUVColor& c, int fw, int fh) {
  if ((unsigned)x >= (unsigned)fw || (unsigned)y >= (unsigned)fh) return;
  yp[y * ys + x] = c.Y;
  uint8_t* uv = uvp + (y >> 1) * us + (x & ~1);
  if (nv21) { uv[0] = c.V; uv[1] = c.U; }
  else      { uv[0] = c.U; uv[1] = c.V; }
}

// Fill a solid-color rectangle.
static void yuvFillRect(uint8_t* yp, uint8_t* uvp, int ys, int us, bool nv21,
                         int x1, int y1, int x2, int y2, const YUVColor& c,
                         int fw, int fh) {
  x1 = std::max(0, x1); y1 = std::max(0, y1);
  x2 = std::min(fw, x2); y2 = std::min(fh, y2);
  if (x1 >= x2 || y1 >= y2) return;

  for (int row = y1; row < y2; ++row)
    std::memset(yp + row * ys + x1, c.Y, x2 - x1);

  int ux1 = x1 & ~1, ux2 = (x2 + 1) & ~1;
  for (int row = y1 >> 1; row < (y2 + 1) >> 1; ++row) {
    uint8_t* uv = uvp + row * us + ux1;
    for (int col = ux1; col < ux2; col += 2) {
      if (nv21) { *uv++ = c.V; *uv++ = c.U; }
      else      { *uv++ = c.U; *uv++ = c.V; }
    }
  }
}

// Draw a rectangle outline with given thickness.
static void yuvDrawRect(uint8_t* yp, uint8_t* uvp, int ys, int us, bool nv21,
                         int x1, int y1, int x2, int y2, int t, const YUVColor& c,
                         int fw, int fh) {
  yuvFillRect(yp, uvp, ys, us, nv21, x1,   y1,   x2,     y1+t,   c, fw, fh); // top
  yuvFillRect(yp, uvp, ys, us, nv21, x1,   y2-t, x2,     y2,     c, fw, fh); // bottom
  yuvFillRect(yp, uvp, ys, us, nv21, x1,   y1+t, x1+t,   y2-t,   c, fw, fh); // left
  yuvFillRect(yp, uvp, ys, us, nv21, x2-t, y1+t, x2,     y2-t,   c, fw, fh); // right
}

// Draw a line using Bresenham's algorithm.
static void yuvDrawLine(uint8_t* yp, uint8_t* uvp, int ys, int us, bool nv21,
                         int x0, int y0, int x1, int y1, const YUVColor& c,
                         int fw, int fh) {
  int dx = std::abs(x1-x0), sx = x0 < x1 ? 1 : -1;
  int dy = -std::abs(y1-y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    yuvSetPixel(yp, uvp, ys, us, nv21, x0, y0, c, fw, fh);
    // Draw 2nd pixel for ~2px thickness on steep lines
    yuvSetPixel(yp, uvp, ys, us, nv21, x0+1, y0, c, fw, fh);
    yuvSetPixel(yp, uvp, ys, us, nv21, x0, y0+1, c, fw, fh);
    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 >= dy) { if (x0 == x1) break; err += dy; x0 += sx; }
    if (e2 <= dx) { if (y0 == y1) break; err += dx; y0 += sy; }
  }
}

// Draw a filled circle.
static void yuvDrawCircle(uint8_t* yp, uint8_t* uvp, int ys, int us, bool nv21,
                            int cx, int cy, int r, const YUVColor& c, int fw, int fh) {
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx)
      if (dx*dx + dy*dy <= r*r)
        yuvSetPixel(yp, uvp, ys, us, nv21, cx+dx, cy+dy, c, fw, fh);
}

// Render text glyphs via OpenCV (grayscale mask) and blit Y-channel only.
// UV is left unchanged — use yuvFillRect to set the background color first.
// fg_Y: luma of the text colour (16 = near-black, 235 = near-white).
// bg_Y: luma of the background (for anti-alias blending).
static void yuvBlitText(uint8_t* yp, int ys, const char* text,
                         int x, int y, double scale, uint8_t fg_Y, uint8_t bg_Y,
                         int fw, int fh) {
  int baseline = 0;
  cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
  cv::Mat glyph(ts.height + baseline + 1, ts.width + 2, CV_8UC1, cv::Scalar(0));
  cv::putText(glyph, text, cv::Point(0, ts.height),
              cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255), 1, cv::LINE_AA);

  for (int row = 0; row < glyph.rows; ++row) {
    int fy = y - ts.height + row;
    if ((unsigned)fy >= (unsigned)fh) continue;
    for (int col = 0; col < glyph.cols; ++col) {
      int fx = x + col;
      if ((unsigned)fx >= (unsigned)fw) continue;
      uint8_t a = glyph.at<uint8_t>(row, col);
      if (a < 16) continue;
      yp[fy * ys + fx] = (uint8_t)(((255 - a) * (int)bg_Y + a * (int)fg_Y) >> 8);
    }
  }
}

// ─── PyRTSP ──────────────────────────────────────────────────────────────────

PyRTSP::PyRTSP(int32_t width, int32_t height, int32_t chn,
               const std::string& codec, const std::string& session_name,
               int32_t bitrate, int32_t gop, int32_t fps) {
  PAYLOAD_TYPE_E payload;
  if (codec == "h265" || codec == "H265")      payload = PT_H265;
  else if (codec == "h264" || codec == "H264") payload = PT_H264;
  else throw std::runtime_error("RTSPServer: codec must be 'h264' or 'h265'");

  session_name_ = session_name.empty()
                      ? (codec == "h265" ? "h265" : "h264")
                      : session_name;

  rtsp_ = std::make_unique<RTSP>(chn, payload, width, height, session_name_,
                                 bitrate, gop, fps);
  LOGI("[PyRTSP] started  chn=%d %dx%d codec=%s  bitrate=%dkbps gop=%d fps=%d  url=rtsp://<ip>:554/%s\n",
       chn, width, height, codec.c_str(), bitrate, gop, fps, session_name_.c_str());
}

PyRTSP::~PyRTSP() { rtsp_.reset(); }

void PyRTSP::sendFrame(const PyImage& image) {
  VPSSImage* vpss = requireVPSS(image, "RTSPServer.send_frame");
  VIDEO_FRAME_INFO_S* frame = vpss->getFrame();
  if (!frame) throw std::runtime_error("RTSPServer.send_frame: null frame");
  int ret = rtsp_->sendFrame(frame);
  if (ret != 0) LOGE("[PyRTSP] sendFrame failed: %d\n", ret);
}

// ─── Draw utilities ──────────────────────────────────────────────────────────

void drawBbox(PyImage& image, int x1, int y1, int x2, int y2,
              py::tuple color, int thickness) {
  VPSSImage* vpss = requireVPSS(image, "draw_bbox");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);
  YUVColor c = parseColorYUV(color);

  PlaneMapping m = mapPlanes(vpss);
  yuvDrawRect(m.y_va, m.uv_va, st[0], st[1], nv21, x1, y1, x2, y2, thickness, c, fw, fh);
  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

void drawText(PyImage& image, const std::string& text, int x, int y,
              py::tuple color, double scale) {
  VPSSImage* vpss = requireVPSS(image, "draw_text");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);
  YUVColor c = parseColorYUV(color);

  PlaneMapping m = mapPlanes(vpss);
  // Render text directly; bg_Y=0 so anti-alias blends toward black background
  yuvBlitText(m.y_va, st[0], text.c_str(), x, y, scale, c.Y, 0, fw, fh);
  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

void drawDetections(PyImage& image, const py::list& detections,
                    float score_threshold) {
  if (detections.empty()) return;
  VPSSImage* vpss = requireVPSS(image, "draw_detections");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  // Map once; flush once at the end. No IonInvalidateCache needed because
  // we only write (not read) the hardware buffer.
  PlaneMapping m = mapPlanes(vpss);

  for (auto item : detections) {
    py::dict det = item.cast<py::dict>();
    float score = det["score"].cast<float>();
    if (score < score_threshold) continue;

    int x1 = std::max(0,    (int)det["x1"].cast<float>());
    int y1 = std::max(0,    (int)det["y1"].cast<float>());
    int x2 = std::min(fw-1, (int)det["x2"].cast<float>());
    int y2 = std::min(fh-1, (int)det["y2"].cast<float>());

    std::string cls_name;
    if (det.contains("class_name")) cls_name = det["class_name"].cast<std::string>();
    int cls_id = det.contains("class_id") ? det["class_id"].cast<int>() : 0;
    YUVColor col = detectionColor(cls_id);

    // Bounding box outline
    yuvDrawRect(m.y_va, m.uv_va, ys, us, nv21, x1, y1, x2, y2, 2, col, fw, fh);

    // Label: measure text first
    char buf[64];
    snprintf(buf, sizeof(buf), "%s %.2f", cls_name.c_str(), score);
    int baseline = 0;
    cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);

    int lx = x1;
    int ly = std::max(ts.height + baseline, y1 - 2);  // top of label box

    // Filled label background (same color as bbox)
    yuvFillRect(m.y_va, m.uv_va, ys, us, nv21,
                lx, ly - ts.height - baseline,
                lx + ts.width + 2, ly + 2,
                col, fw, fh);

    // White text on dark backgrounds (Y<128), black on light backgrounds
    uint8_t text_Y = (col.Y < 128) ? 235 : 16;
    yuvBlitText(m.y_va, ys, buf, lx + 1, ly, 0.4, text_Y, col.Y, fw, fh);
  }

  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// COCO-17 skeleton connectivity (19 pairs, matches C++ sample_img_human_keypoint.cpp)
static const int kCOCO17Skeleton[][2] = {
    {15,13},{13,11},{16,14},{14,12},{11,12},
    {5,11}, {6,12}, {5,6},  {5,7},  {6,8},
    {7,9},  {8,10}, {1,2},  {0,1},  {0,2},
    {1,3},  {2,4},  {3,5},  {4,6}
};
static const int kCOCO17SkeletonLen =
    (int)(sizeof(kCOCO17Skeleton) / sizeof(kCOCO17Skeleton[0]));

void drawKeypoints(PyImage& image, const py::list& detections_with_landmarks,
                   float score_threshold) {
  if (detections_with_landmarks.empty()) return;
  VPSSImage* vpss = requireVPSS(image, "draw_keypoints");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  static const YUVColor kBone   = YUVColor::fromRGB(  0, 128, 255); // orange-blue
  static const YUVColor kJoint  = YUVColor::fromRGB(  0, 255, 128); // green

  PlaneMapping m = mapPlanes(vpss);

  for (auto item : detections_with_landmarks) {
    py::dict det = item.cast<py::dict>();

    std::vector<float> lx, ly, ls;

    if (det.contains("landmarks_x") && det.contains("landmarks_y")) {
      // Format A: separate flat lists  {"landmarks_x": [...], "landmarks_y": [...]}
      lx = det["landmarks_x"].cast<std::vector<float>>();
      ly = det["landmarks_y"].cast<std::vector<float>>();
    } else if (det.contains("landmarks")) {
      // Format B: list of [x,y] pairs  {"landmarks": [[x0,y0], [x1,y1], ...]}
      for (auto pt_obj : det["landmarks"].cast<py::list>()) {
        auto pt = pt_obj.cast<py::list>();
        lx.push_back(pt[0].cast<float>());
        ly.push_back(pt[1].cast<float>());
      }
    } else {
      continue;
    }

    // landmarks_score: accept either a flat list or a single float
    if (det.contains("landmarks_score")) {
      py::object lso = det["landmarks_score"];
      try {
        ls = lso.cast<std::vector<float>>();
      } catch (...) {
        ls.assign(lx.size(), lso.cast<float>());
      }
    }

    // Overall confidence: use "score" if present, otherwise landmarks_score[0].
    // For OBJECT_LANDMARKS models (e.g. KEYPOINT_FACE_V2) there is no "score"
    // key — landmarks_score[0] is the face presence confidence.
    float score = 1.0f;
    if (det.contains("score")) {
      score = det["score"].cast<float>();
    } else if (!ls.empty()) {
      score = ls[0];
    }
    if (score < score_threshold) continue;

    int n = (int)std::min(lx.size(), ly.size());
    if (n == 0) continue;

    // Auto-detect normalized coordinates [0.0, 1.0].
    // Models like KEYPOINT_HAND output normalized values; pose/face use pixels.
    // Heuristic: if every x AND y value is in [0, 1] the coords are normalized.
    {
      float mx = *std::max_element(lx.begin(), lx.end());
      float my = *std::max_element(ly.begin(), ly.end());
      if (mx <= 1.0f && my <= 1.0f) {
        for (auto& v : lx) v *= fw;
        for (auto& v : ly) v *= fh;
      }
    }

    // Only use per-landmark scores when the scores vector matches the number
    // of landmarks.  Some models (e.g. KEYPOINT_FACE_V2) return a smaller
    // scores vector (face score + blurness) that must NOT be used as a
    // per-landmark filter.
    bool use_per_pt_score = ((int)ls.size() == n);

    // Lane detection: exactly 2 points → draw a line segment, not dots.
    static const YUVColor kLane = YUVColor::fromRGB(0, 220, 0); // green
    if (n == 2) {
      yuvDrawLine(m.y_va, m.uv_va, ys, us, nv21,
                  (int)lx[0], (int)ly[0], (int)lx[1], (int)ly[1],
                  kLane, fw, fh);
      continue;
    }

    // Skeleton lines (COCO-17 pose, 19 connections)
    if (n == 17) {
      for (int k = 0; k < kCOCO17SkeletonLen; ++k) {
        int a = kCOCO17Skeleton[k][0], b = kCOCO17Skeleton[k][1];
        if (a >= n || b >= n) continue;
        if (use_per_pt_score && ls[a] < 0.3f) continue;
        if (use_per_pt_score && ls[b] < 0.3f) continue;
        yuvDrawLine(m.y_va, m.uv_va, ys, us, nv21,
                    (int)lx[a], (int)ly[a], (int)lx[b], (int)ly[b],
                    kBone, fw, fh);
      }
    }

    // Joint dots — radius scales with image height for visibility
    int dot_r = std::max(3, fh / 180);
    for (int k = 0; k < n; ++k) {
      if (use_per_pt_score && ls[k] < 0.3f) continue;
      yuvDrawCircle(m.y_va, m.uv_va, ys, us, nv21,
                    (int)lx[k], (int)ly[k], dot_r, kJoint, fw, fh);
    }
  }

  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// ─── Draw Classification ─────────────────────────────────────────────────────
//
// Draws a text label for CLASSIFICATION and CLS_ATTRIBUTE outputs.
// result: a list or dict as returned by model.inference().
// Renders a filled label box in the top-left corner of the frame.

void drawClassification(PyImage& image, const py::object& result) {
  VPSSImage* vpss = requireVPSS(image, "draw_classification");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  py::dict det;
  if (py::isinstance<py::list>(result)) {
    py::list lst = result.cast<py::list>();
    if (lst.empty()) return;
    if (!py::isinstance<py::dict>(lst[0])) return;
    det = lst[0].cast<py::dict>();
  } else if (py::isinstance<py::dict>(result)) {
    det = result.cast<py::dict>();
  } else {
    return;
  }

  // Build list of lines to display
  std::vector<std::string> lines;
  char buf[256];

  // ── CLS_ATTRIBUTE: iterate dynamically over all *_score keys ──────────────
  // Display order follows kAttrOrder; any unknown attr_* keys are appended.
  static const std::vector<std::string> kAttrOrder = {
      "gender", "age", "glasses", "mask", "hat", "emotion", "pose", "blurness"};

  bool is_attr = false;
  for (const auto& name : kAttrOrder) {
    std::string score_key = name + "_score";
    if (!det.contains(score_key.c_str())) continue;
    is_attr = true;
    float score = det[score_key.c_str()].cast<float>();

    if (name == "gender") {
      const char* v = (det.contains("is_male") && det["is_male"].cast<bool>())
                      ? "Male" : "Female";
      snprintf(buf, sizeof(buf), "Gender : %s  %.0f%%", v, score * 100.f);
    } else if (name == "age") {
      int age = det.contains("age") ? det["age"].cast<int>()
                                    : static_cast<int>(score * 100.f);
      snprintf(buf, sizeof(buf), "Age    : %d", age);
    } else if (name == "glasses") {
      const char* v = (det.contains("is_wearing_glasses") &&
                       det["is_wearing_glasses"].cast<bool>()) ? "Yes" : "No";
      snprintf(buf, sizeof(buf), "Glasses: %s  %.0f%%", v, score * 100.f);
    } else if (name == "mask") {
      const char* v = (det.contains("is_wearing_mask") &&
                       det["is_wearing_mask"].cast<bool>()) ? "Yes" : "No";
      snprintf(buf, sizeof(buf), "Mask   : %s  %.0f%%", v, score * 100.f);
    } else if (name == "hat") {
      const char* v = (det.contains("is_wearing_hat") &&
                       det["is_wearing_hat"].cast<bool>()) ? "Yes" : "No";
      snprintf(buf, sizeof(buf), "Hat    : %s  %.0f%%", v, score * 100.f);
    } else {
      // emotion / pose / blurness: show label + raw score
      std::string label = name;
      label[0] = static_cast<char>(toupper(static_cast<unsigned char>(label[0])));
      snprintf(buf, sizeof(buf), "%-7s: %.2f", label.c_str(), score);
    }
    lines.emplace_back(buf);
  }
  // Any attr_<id>_score keys not in kAttrOrder (future enum values)
  if (is_attr) {
    for (auto item : det) {
      std::string key = item.first.cast<std::string>();
      if (key.size() > 11 &&
          key.substr(0, 5) == "attr_" &&
          key.substr(key.size() - 6) == "_score") {
        float score = item.second.cast<float>();
        snprintf(buf, sizeof(buf), "%-7s: %.2f", key.c_str(), score);
        lines.emplace_back(buf);
      }
    }
  }

  // ── CLASSIFICATION: class_name (injected by Python) + score ───────────────
  if (!is_attr) {
    if (det.contains("class_name") && det.contains("score")) {
      snprintf(buf, sizeof(buf), "%s  %.0f%%",
               det["class_name"].cast<std::string>().c_str(),
               det["score"].cast<float>() * 100.0f);
      lines.emplace_back(buf);
    } else if (det.contains("class_id") && det.contains("score")) {
      // Fallback when Python didn't inject class_name
      snprintf(buf, sizeof(buf), "cls%d  %.0f%%",
               det["class_id"].cast<int>(),
               det["score"].cast<float>() * 100.0f);
      lines.emplace_back(buf);
    } else {
      return;
    }
  }

  if (lines.empty()) return;

  // Measure all lines to find bounding box
  const double scale   = 0.45;
  const int    padding = 5;
  const int    line_gap = 4;
  int baseline = 0;
  int max_w = 0, line_h = 0;
  for (const auto& l : lines) {
    cv::Size ts = cv::getTextSize(l.c_str(), cv::FONT_HERSHEY_SIMPLEX,
                                  scale, 1, &baseline);
    if (ts.width > max_w) max_w = ts.width;
    if (ts.height > line_h) line_h = ts.height;
  }
  int n       = (int)lines.size();
  int bx1     = 8, by1 = 8;
  int bx2     = bx1 + max_w + padding * 2;
  int by2     = by1 + n * (line_h + line_gap) + baseline + padding;

  int cls_id  = det.contains("class_id") ? det["class_id"].cast<int>() : 0;
  YUVColor col = detectionColor(cls_id);

  PlaneMapping m   = mapPlanes(vpss);
  uint8_t text_Y   = (col.Y < 128) ? 235 : 16;

  yuvFillRect(m.y_va, m.uv_va, ys, us, nv21, bx1, by1, bx2, by2, col, fw, fh);

  for (int i = 0; i < n; ++i) {
    int ty = by1 + padding + (i + 1) * (line_h + line_gap) - line_gap;
    yuvBlitText(m.y_va, ys, lines[i].c_str(),
                bx1 + padding, ty, scale, text_Y, col.Y, fw, fh);
  }

  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// ─── Draw Segmentation ────────────────────────────────────────────────────────
//
// Draws a semantic segmentation overlay (per-pixel color tint).
// result: a list/dict with keys "output_width", "output_height", "class_id"
// alpha: blend factor 0.0 (invisible) … 1.0 (opaque).  Default 0.5.
// class_id == 0 is treated as background and left untouched.

void drawSegmentation(PyImage& image, const py::object& result, float alpha) {
  VPSSImage* vpss = requireVPSS(image, "draw_segmentation");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  py::dict seg;
  if (py::isinstance<py::list>(result)) {
    py::list lst = result.cast<py::list>();
    if (lst.empty()) return;
    seg = lst[0].cast<py::dict>();
  } else {
    seg = result.cast<py::dict>();
  }

  if (!seg.contains("output_width") || !seg.contains("class_id")) return;
  int sw = seg["output_width"].cast<int>();
  int sh = seg["output_height"].cast<int>();
  auto class_ids = seg["class_id"].cast<std::vector<int>>();
  if ((int)class_ids.size() < sw * sh) return;

  PlaneMapping m = mapPlanes(vpss);
  CVI_SYS_IonInvalidateCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonInvalidateCache(m.uv_pa, m.uv_va, m.uv_len);

  int ia = (int)(alpha * 256.0f + 0.5f);
  int ib = 256 - ia;

  for (int fy = 0; fy < fh; ++fy) {
    int sy = fy * sh / fh;
    for (int fx = 0; fx < fw; ++fx) {
      int sx = fx * sw / fw;
      int cls = class_ids[sy * sw + sx];
      if (cls == 0) continue;
      YUVColor c = detectionColor(cls - 1);
      m.y_va[fy * ys + fx] =
          (uint8_t)((ib * m.y_va[fy * ys + fx] + ia * c.Y) >> 8);
      if ((fy & 1) == 0 && (fx & 1) == 0) {
        uint8_t* uv = m.uv_va + (fy >> 1) * us + (fx & ~1);
        if (nv21) {
          uv[0] = (uint8_t)((ib * uv[0] + ia * c.V) >> 8);
          uv[1] = (uint8_t)((ib * uv[1] + ia * c.U) >> 8);
        } else {
          uv[0] = (uint8_t)((ib * uv[0] + ia * c.U) >> 8);
          uv[1] = (uint8_t)((ib * uv[1] + ia * c.V) >> 8);
        }
      }
    }
  }

  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// ─── Draw Instance Segmentation ───────────────────────────────────────────────
//
// Draws bounding boxes + per-instance mask overlay.
// result: a list/dict with keys "mask_width", "mask_height", "bboxes_seg"
// alpha: mask blend factor (default 0.45).

void drawInstanceSegmentation(PyImage& image, const py::object& result,
                               float score_threshold, float alpha) {
  VPSSImage* vpss = requireVPSS(image, "draw_instance_segmentation");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  py::dict inst;
  if (py::isinstance<py::list>(result)) {
    py::list lst = result.cast<py::list>();
    if (lst.empty()) return;
    inst = lst[0].cast<py::dict>();
  } else {
    inst = result.cast<py::dict>();
  }

  if (!inst.contains("bboxes_seg")) return;
  int mw = inst["mask_width"].cast<int>();
  int mh = inst["mask_height"].cast<int>();
  auto bboxes = inst["bboxes_seg"].cast<py::list>();

  PlaneMapping m = mapPlanes(vpss);
  CVI_SYS_IonInvalidateCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonInvalidateCache(m.uv_pa, m.uv_va, m.uv_len);

  int ia = (int)(alpha * 256.0f + 0.5f);
  int ib = 256 - ia;

  for (auto item : bboxes) {
    py::dict det = item.cast<py::dict>();
    float score = det.contains("score") ? det["score"].cast<float>() : 1.0f;
    if (score < score_threshold) continue;

    int cls_id = det.contains("class_id") ? det["class_id"].cast<int>() : 0;
    YUVColor col = detectionColor(cls_id);

    int x1 = std::max(0,    (int)det["x1"].cast<float>());
    int y1 = std::max(0,    (int)det["y1"].cast<float>());
    int x2 = std::min(fw-1, (int)det["x2"].cast<float>());
    int y2 = std::min(fh-1, (int)det["y2"].cast<float>());

    // Bounding box
    yuvDrawRect(m.y_va, m.uv_va, ys, us, nv21, x1, y1, x2, y2, 2, col, fw, fh);

    // Label
    if (det.contains("class_name")) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%s %.2f",
               det["class_name"].cast<std::string>().c_str(), score);
      int baseline = 0;
      cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
      int lx = x1;
      int ly = std::max(ts.height + baseline, y1 - 2);
      yuvFillRect(m.y_va, m.uv_va, ys, us, nv21,
                  lx, ly - ts.height - baseline, lx + ts.width + 2, ly + 2,
                  col, fw, fh);
      uint8_t text_Y = (col.Y < 128) ? 235 : 16;
      yuvBlitText(m.y_va, ys, buf, lx + 1, ly, 0.4, text_Y, col.Y, fw, fh);
    }

    // Mask overlay (scaled from mask space to bbox space)
    if (det.contains("mask") && mw > 0 && mh > 0) {
      std::vector<float> mask_vals;
      try {
        mask_vals = det["mask"].cast<std::vector<float>>();
      } catch (...) {
        auto ml = det["mask"].cast<py::list>();
        mask_vals.reserve(mw * mh);
        for (auto v : ml)
          try { mask_vals.push_back(v.cast<float>()); }
          catch (...) { mask_vals.push_back((float)v.cast<int>()); }
      }
      if ((int)mask_vals.size() < mw * mh) continue;

      int bw = x2 - x1, bh = y2 - y1;
      if (bw <= 0 || bh <= 0) continue;

      for (int fy = y1; fy < y2; ++fy) {
        int my = (fy - y1) * mh / bh;
        for (int fx = x1; fx < x2; ++fx) {
          int mx = (fx - x1) * mw / bw;
          if (mask_vals[my * mw + mx] < 0.5f) continue;
          m.y_va[fy * ys + fx] =
              (uint8_t)((ib * m.y_va[fy * ys + fx] + ia * col.Y) >> 8);
          if ((fy & 1) == 0 && (fx & 1) == 0) {
            uint8_t* uv = m.uv_va + (fy >> 1) * us + (fx & ~1);
            if (nv21) {
              uv[0] = (uint8_t)((ib * uv[0] + ia * col.V) >> 8);
              uv[1] = (uint8_t)((ib * uv[1] + ia * col.U) >> 8);
            } else {
              uv[0] = (uint8_t)((ib * uv[0] + ia * col.U) >> 8);
              uv[1] = (uint8_t)((ib * uv[1] + ia * col.V) >> 8);
            }
          }
        }
      }
    }
  }

  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// ─── Draw OCR ─────────────────────────────────────────────────────────────────
//
// Draws OCR text result at the bottom of the frame.
// result: a list containing a string, as returned by OCR models.

void drawOcr(PyImage& image, const py::object& result) {
  VPSSImage* vpss = requireVPSS(image, "draw_ocr");
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  std::string text;
  if (py::isinstance<py::list>(result)) {
    py::list lst = result.cast<py::list>();
    if (lst.empty()) return;
    try { text = lst[0].cast<std::string>(); } catch (...) { return; }
  } else if (py::isinstance<py::str>(result)) {
    text = result.cast<std::string>();
  } else {
    return;
  }
  if (text.empty()) return;

  static const YUVColor kBg = YUVColor::fromRGB(0, 0, 0);

  double scale = 0.6;
  int margin = 8;
  int baseline = 0;
  cv::Size ts = cv::getTextSize(text.c_str(), cv::FONT_HERSHEY_SIMPLEX,
                                 scale, 1, &baseline);
  int bx1 = margin;
  int by2 = fh - margin;
  int bx2 = std::min(fw - margin, bx1 + ts.width + 6);
  int by1 = by2 - ts.height - baseline - 6;

  PlaneMapping m = mapPlanes(vpss);
  yuvFillRect(m.y_va, m.uv_va, ys, us, nv21, bx1, by1, bx2, by2, kBg, fw, fh);
  yuvBlitText(m.y_va, ys, text.c_str(),
              bx1 + 3, by2 - baseline - 3, scale, 235, 16, fw, fh);
  CVI_SYS_IonFlushCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonFlushCache(m.uv_pa, m.uv_va, m.uv_len);
}

// ─── JPEG export ─────────────────────────────────────────────────────────────
//
// Convert a hardware camera frame (YUV420SP) to JPEG and return Python bytes.
// Caller must hold the frame (i.e. not yet cam.release()'d).
// quality: 0-100 JPEG quality (default 80).

py::bytes frameToJpeg(const PyImage& image, int quality, float scale) {
  requireVPSS(image, "frame_to_jpeg");

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)

  // ── Hardware path (scale = 1.0 only) ────────────────────────────────────
  // CVI_VENC encodes YUV420SP directly — no YUV→BGR conversion on the CPU.
  // When scale < 1 we fall through to the SW path, which encodes at the
  // smaller resolution directly (faster than HW encode + SW decode + resize).
  if (scale >= 1.0f) {
    std::vector<uint8_t> hw_buf;
    if (hwJpegEncoder().encodeFrame(image.getImage(), hw_buf, 1, quality)) {
      return py::bytes(reinterpret_cast<const char*>(hw_buf.data()), hw_buf.size());
    }
  }
  // Fall through to software path (scale < 1 or HW encoder failure).

#endif

  // ── Software fallback (non-chip platforms or HW failure) ─────────────────
  VPSSImage* vpss = dynamic_cast<VPSSImage*>(image.getImage().get());
  int fh = (int)vpss->getHeight(), fw = (int)vpss->getWidth();
  auto st = vpss->getStrides();
  int ys = (int)st[0], us = (int)st[1];
  bool nv21 = (vpss->getImageFormat() == ImageFormat::YUV420SP_VU);

  PlaneMapping m = mapPlanes(vpss);
  CVI_SYS_IonInvalidateCache(m.y_pa, m.y_va, m.y_len);
  CVI_SYS_IonInvalidateCache(m.uv_pa, m.uv_va, m.uv_len);

  cv::Mat yuv(fh + fh / 2, fw, CV_8UC1);
  for (int row = 0; row < fh; ++row)
    std::memcpy(yuv.ptr(row), m.y_va + row * ys, fw);
  for (int row = 0; row < fh / 2; ++row)
    std::memcpy(yuv.ptr(fh + row), m.uv_va + row * us, fw);

  cv::Mat bgr;
  cv::cvtColor(yuv, bgr, nv21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12);

  if (scale > 0.0f && scale < 1.0f) {
    cv::resize(bgr, bgr,
               cv::Size((int)(fw * scale), (int)(fh * scale)),
               0, 0, cv::INTER_LINEAR);
  }

  std::vector<uint8_t> buf;
  cv::imencode(".jpg", bgr, buf, {cv::IMWRITE_JPEG_QUALITY, quality});
  return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
}

}  // namespace pytdl
