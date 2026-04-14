#include "py_rtsp_client.hpp"

#include <stdexcept>
#include <string>

#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"

namespace pytdl {

PyRtspClient::PyRtspClient(const std::string& url, int width, int height,
                           int timeout_ms, const std::string& transport) {
  target_width_  = width;
  target_height_ = height;

  // Build GStreamer/FFmpeg options for OpenCV VideoCapture.
  // opencv_ffmpeg_capture_options is set via cv::CAP_PROP_* or the env var
  // OPENCV_FFMPEG_CAPTURE_OPTIONS before open().  We use the CAP_PROP approach.
  cap_.open(url, cv::CAP_FFMPEG);
  if (!cap_.isOpened()) {
    throw std::runtime_error("RtspClient: failed to open stream: " + url);
  }

  // Set transport and timeout (best-effort — silently ignored if not supported).
  // RTSP_TRANSPORT: "tcp" (default) is more reliable over Wi-Fi.
  cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms);
  cap_.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms);

  int actual_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

  LOGI("[RtspClient] opened %s  native=%dx%d  target=%dx%d  transport=%s\n",
       url.c_str(), actual_w, actual_h,
       (target_width_  ? target_width_  : actual_w),
       (target_height_ ? target_height_ : actual_h),
       transport.c_str());
}

PyRtspClient::~PyRtspClient() { close(); }

PyImage PyRtspClient::read() {
  if (closed_) throw std::runtime_error("RtspClient: stream is closed");

  cv::Mat bgr;
  if (!cap_.read(bgr) || bgr.empty()) {
    throw std::runtime_error("RtspClient: failed to read frame (end of stream or timeout)");
  }

  // Optionally resize to the requested resolution.
  if (target_width_ > 0 && target_height_ > 0) {
    if (bgr.cols != target_width_ || bgr.rows != target_height_) {
      cv::resize(bgr, bgr, cv::Size(target_width_, target_height_));
    }
  }

  const uint32_t w = static_cast<uint32_t>(bgr.cols);
  const uint32_t h = static_cast<uint32_t>(bgr.rows);

  // Convert BGR → YUV420 NV12 so the frame is compatible with both the
  // hardware VENC encoder (RTSPServer.send_frame) and the VPSS preprocessor
  // used during inference.  CVI VENC does NOT accept BGR input directly.
  //
  // OpenCV's COLOR_BGR2YUV_I420 produces a (h*3/2) × w Mat:
  //   rows [0,   h)        → Y  plane,  full width
  //   rows [h,   h+h/4)    → U  plane,  half width  (I420 planar)
  //   rows [h+h/4, h+h/2)  → V  plane,  half width
  // We then interleave U and V to produce NV12 (UVUVUV…).
  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

  // Allocate a VPSSImage (ION/CMA memory) in NV12 format.
  auto vpss = std::make_shared<VPSSImage>(w, h, ImageFormat::YUV420SP_UV,
                                          TDLDataType::UINT8,
                                          /*alloc_memory=*/true);

  std::vector<uint8_t*> dst_ptrs = vpss->getVirtualAddress();
  std::vector<uint32_t> strides  = vpss->getStrides();

  if (dst_ptrs.size() < 2 || dst_ptrs[0] == nullptr || dst_ptrs[1] == nullptr) {
    throw std::runtime_error("RtspClient: failed to map VPSSImage virtual address");
  }

  // ── Y plane ──────────────────────────────────────────────────────────────
  const uint32_t y_dst_stride = strides[0];
  uint8_t*       y_dst        = dst_ptrs[0];
  const uint8_t* y_src        = i420.data;            // rows [0, h)
  for (uint32_t row = 0; row < h; ++row)
    std::memcpy(y_dst + row * y_dst_stride, y_src + row * w, w);

  // ── UV interleaved plane (NV12 = UVUV…) ──────────────────────────────────
  const uint32_t uv_dst_stride = strides[1];
  uint8_t*       uv_dst        = dst_ptrs[1];
  const uint8_t* u_src = i420.data + static_cast<size_t>(h) * w;        // U plane
  const uint8_t* v_src = i420.data + static_cast<size_t>(h) * w * 5/4;  // V plane
  const uint32_t half_w = w / 2;
  const uint32_t half_h = h / 2;
  for (uint32_t row = 0; row < half_h; ++row) {
    uint8_t*       d = uv_dst + row * uv_dst_stride;
    const uint8_t* u = u_src  + row * half_w;
    const uint8_t* v = v_src  + row * half_w;
    for (uint32_t col = 0; col < half_w; ++col) {
      d[col * 2]     = u[col];  // U
      d[col * 2 + 1] = v[col];  // V
    }
  }

  // Flush CPU cache so the VPSS hardware sees the written data.
  vpss->flushCache();

  std::shared_ptr<BaseImage> base = vpss;
  return PyImage(base);
}

void PyRtspClient::close() {
  if (!closed_) {
    cap_.release();
    closed_ = true;
  }
}

bool PyRtspClient::isOpened() const {
  return !closed_ && cap_.isOpened();
}

}  // namespace pytdl
