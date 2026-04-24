#ifndef PYTHON_USB_CAMERA_HPP_
#define PYTHON_USB_CAMERA_HPP_

#include <pybind11/pybind11.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <cvi_vb.h>
#include "opencv2/opencv.hpp"
#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

// USB camera capture via V4L2 (OpenCV VideoCapture backend).
//
// Frames are returned as VPSSImage backed by VB pool memory, making them
// fully compatible with CVI_VENC, RTSPServer.send_frame(),
// model.inference(), draw_detections(), and all other TDL/CVI functions.
//
// IMPORTANT: CVI_VENC requires input frames to reside in VB pool memory
// (with a valid u32PoolId).  Raw ION memory (u32PoolId=0) causes the
// RTSP stream to freeze after the first I-frame because P-frames
// cannot reference frames outside a VB pool.
//
// A background thread continuously prefetches the next frame so that
// camera capture overlaps with Python-side inference/encode, roughly
// doubling throughput when both take similar time.
//
// Usage:
//   cam = image.UsbCamera(0, 640, 480)   # device index, width, height
//   frame = cam.read()
//   result = model.inference(frame)
//   cam.release()
//   cam.close()
class PyUsbCamera {
 public:
  // device: V4L2 device index (0 = /dev/video0, 1 = /dev/video1, ...)
  // width / height: requested capture resolution (0 = camera default)
  explicit PyUsbCamera(int device = 0, int width = 640, int height = 480);
  ~PyUsbCamera();

  // Return the next frame as a PyImage (VPSSImage NV21, VB pool memory).
  // If the prefetch thread already captured a frame, this returns
  // immediately; otherwise it blocks until one is ready.
  // Raises RuntimeError on failure.
  PyImage read();

  // No-op. Provided for API compatibility with Camera and RtspClient.
  void release() {}

  // Close the device and free resources.
  void close();

  bool isOpened() const;

  // Actual capture resolution (may differ from requested).
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }

  // Context manager support.
  PyUsbCamera* enter() { return this; }
  void exit(py::object, py::object, py::object) { close(); }

 private:
  // A captured frame living in VB pool memory, not yet wrapped in PyImage.
  struct PrefetchedFrame {
    VIDEO_FRAME_INFO_S frame_info;
    VB_BLK blk;
    void* vir_base;
    uint32_t mapped_size;
  };

  // Release a PrefetchedFrame's VB resources.
  static void releasePrefetched(PrefetchedFrame* pf);

  // Capture one frame from the camera into a VB block.
  // Returns nullptr if the camera fails to deliver a frame.
  std::unique_ptr<PrefetchedFrame> captureOne();

  // Background thread function: continuously prefetches the next frame.
  void captureLoop();

  cv::VideoCapture cap_;
  int width_  = 0;
  int height_ = 0;
  bool native_nv12_ = false;  // true when camera delivers NV12 directly
  bool closed_ = false;

  // Monotonic clock origin for PTS generation (microseconds).
  std::chrono::steady_clock::time_point pts_origin_ =
      std::chrono::steady_clock::now();

  // VB pool for frame buffers.  CVI_VENC requires frames from a VB pool,
  // not raw ION memory.
  VB_POOL vb_pool_ = VB_INVALID_POOLID;
  uint32_t vb_blk_size_ = 0;

  // ── Prefetch thread state ────────────────────────────────────────────────
  std::thread capture_thread_;
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_;
  std::unique_ptr<PrefetchedFrame> prefetched_;  // slot: one ready frame
  std::string capture_error_;                     // set by thread on failure
  bool stop_thread_ = false;
};

}  // namespace pytdl
#endif  // PYTHON_USB_CAMERA_HPP_
