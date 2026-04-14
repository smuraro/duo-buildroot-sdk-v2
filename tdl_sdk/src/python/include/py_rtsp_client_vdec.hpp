#ifndef PYTHON_RTSP_CLIENT_VDEC_HPP_
#define PYTHON_RTSP_CLIENT_VDEC_HPP_

// Hardware-accelerated RTSP client: live555 (RTSP/RTP) → CVITEK VDEC (H264)
//
// H265 note: the CV181X Wave4 VPU does not support H265 decode in practice.
// When an H265 stream is detected the class transparently falls back to
// OpenCV/FFmpeg software decoding so callers need not special-case the codec.
//
// Benefits over RtspClient (OpenCV/FFmpeg software decode) for H264:
//   - H264 decoded entirely by the VDEC hardware unit — zero CPU cost
//   - Decoded frames land in VB memory (same as Camera frames)
//   - Output is YUV420 NV12, processed by VPSS — no BGR copy needed
//   - Lower latency: no FFmpeg multi-frame buffer pipeline
//
// Python usage:
//   client = image.RtspClientVdec("rtsp://192.168.1.10:554/live",
//                                  width=640, height=480)
//   with client:
//       while True:
//           frame  = client.read()
//           result = model.inference(frame)
//           client.release()   # MUST be called before the next read()

#include <pybind11/pybind11.h>

#include <atomic>
#include <future>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

// CVI middleware (needed for VIDEO_FRAME_INFO_S in the field declaration)
#include "cvi_comm_video.h"
#include "cvi_vdec.h"

#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

class PyRtspClientVdec {
 public:
  // url:         RTSP stream URL (rtsp://...)
  // width/height: maximum output resolution; 0 = use native stream resolution
  // timeout_ms:  per-frame timeout (VDEC GetFrame or OpenCV read)
  // transport:   "tcp" (default, reliable) or "udp"
  explicit PyRtspClientVdec(const std::string& url,
                            int width = 0, int height = 0,
                            int timeout_ms = 5000,
                            const std::string& transport = "tcp");
  ~PyRtspClientVdec();

  // Decode the next frame.
  // H264: returns YUV420 NV12 from VDEC hardware.
  // H265: returns YUV420 NV12 converted from OpenCV/FFmpeg software decode.
  // The caller MUST call release() before calling read() again.
  PyImage read();

  // Return the current decoded frame buffer back to the VDEC pool (H264).
  // No-op for H265 OpenCV fallback.
  void release();

  // Async inference support (H264 only; no-op for OpenCV fallback):
  //   pin_for_inference() — moves the current held frame to the inference slot
  //     so the inference thread can safely read it while read() fetches the next
  //     frame.  Must be called AFTER send_frame() (no more writes to the frame)
  //     and before the next read().
  //   release_inference() — releases the inference slot back to the VDEC pool.
  //     Must be called AFTER the inference thread has finished.
  void pinForInference();
  void releaseInference();

  // Stop the stream and release all resources.
  void close();

  bool isOpened() const;

  // Context manager support
  PyRtspClientVdec* enter() { return this; }
  void exit(py::object, py::object, py::object) { close(); }

  // ── Internal API (public to allow access from anonymous-namespace callbacks) ──
  // Initialise the VDEC channel.  Called from within the event-loop thread.
  bool initVdec(uint32_t width, uint32_t height, bool is_h265);
  // Open OpenCV fallback for H265.  Called from within the event-loop thread.
  bool openCvFallback();
  // Signal that setup succeeded or failed.  Called from the event thread.
  void onVdecReady(bool ok);

 private:
  void eventLoopThread();
  void deinitVdec();

  std::string url_;
  std::string transport_;
  int timeout_ms_    = 5000;
  int target_width_  = 0;
  int target_height_ = 0;
  int vdec_chn_      = -1;
  bool closed_       = false;
  bool is_opened_    = false;

  // Signalling: event thread → constructor
  std::promise<bool> ready_promise_;
  std::future<bool>  ready_future_;

  // Set to non-zero to stop the live555 event loop
  char event_watch_ = 0;
  std::thread event_thread_;

  // H265: OpenCV fallback
  bool             use_cv_fallback_ = false;
  cv::VideoCapture cv_cap_;

  // H264: current frame held by the caller (between read() and release())
  bool frame_held_ = false;
  VIDEO_FRAME_INFO_S held_frame_{};

  // H264: inference frame pinned by pin_for_inference() until release_inference()
  bool infer_frame_held_ = false;
  VIDEO_FRAME_INFO_S infer_frame_{};

  // Dedicated VB pools (avoids conflicts with Camera / common pool)
  VB_POOL vb_pool_     = VB_INVALID_POOLID;  // picture (YUV) frames
  VB_POOL vb_tmv_pool_ = VB_INVALID_POOLID;  // H264 co-located MV buffers

  // Global VDEC channel counter
  static std::atomic<int> next_vdec_chn_;
};

}  // namespace pytdl
#endif  // PYTHON_RTSP_CLIENT_VDEC_HPP_
