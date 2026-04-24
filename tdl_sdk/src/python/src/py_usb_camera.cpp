#include "py_usb_camera.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <pthread.h>
#include <sched.h>

#include <cvi_buffer.h>
#include <cvi_vb.h>
#include "cvi_sys.h"
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"

namespace pytdl {

// ---------------------------------------------------------------------------
// VbBackedImage — VPSSImage subclass that releases a VB block on destruction.
//
// CVI_VENC requires input frames to live in VB pool memory (valid u32PoolId).
// This class wraps a VIDEO_FRAME_INFO_S whose physical memory comes from a
// VB block.  When the Python-side PyImage reference count drops to zero,
// this destructor unmaps and releases the VB block automatically.
// ---------------------------------------------------------------------------
class VbBackedImage : public VPSSImage {
 public:
  VbBackedImage(const VIDEO_FRAME_INFO_S& frame, VB_BLK blk,
                void* mapped_vir, uint32_t mapped_size)
      : VPSSImage(frame),
        blk_(blk),
        mapped_vir_(mapped_vir),
        mapped_size_(mapped_size) {}

  ~VbBackedImage() override {
    if (mapped_vir_) {
      CVI_SYS_Munmap(mapped_vir_, mapped_size_);
      mapped_vir_ = nullptr;
    }
    CVI_VB_ReleaseBlock(blk_);
  }

 private:
  VB_BLK blk_;
  void* mapped_vir_;
  uint32_t mapped_size_;
};

// ---------------------------------------------------------------------------
// releasePrefetched — free VB block + mmap for a PrefetchedFrame.
// ---------------------------------------------------------------------------
void PyUsbCamera::releasePrefetched(PrefetchedFrame* pf) {
  if (!pf) return;
  if (pf->vir_base) {
    CVI_SYS_Munmap(pf->vir_base, pf->mapped_size);
    pf->vir_base = nullptr;
  }
  CVI_VB_ReleaseBlock(pf->blk);
}

// ---------------------------------------------------------------------------
// PyUsbCamera implementation
// ---------------------------------------------------------------------------

PyUsbCamera::PyUsbCamera(int device, int width, int height)
    : width_(width), height_(height) {
  // Prevent OpenCV from spawning extra threads for cvtColor/resize.
  // On single-core SoCs, extra threads add overhead without benefit.
  cv::setNumThreads(1);

  cap_.open(device, cv::CAP_V4L2);
  if (!cap_.isOpened()) {
    throw std::runtime_error("UsbCamera: failed to open /dev/video" +
                             std::to_string(device));
  }

  // Request the desired resolution (best-effort; camera may round to nearest).
  if (width_ > 0)  cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
  if (height_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);

  // Read actual dimensions BEFORE changing the pixel format, because
  // CAP_PROP_FORMAT = -1 can change the reported width/height on some
  // V4L2 drivers (e.g. YUYV width doubles to byte-width).
  int actual_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
  width_  = actual_w;
  height_ = actual_h;

  // Check if the camera natively outputs NV12.
  // CAP_PROP_FORMAT = -1 reliably delivers raw NV12 frames, but does NOT
  // work for YUYV on many V4L2/OpenCV combos (returns BGR despite the
  // flag).  For YUYV cameras we let OpenCV auto-convert to BGR and then
  // convert BGR → I420 → NV21 in the fallback path.
  double fourcc = cap_.get(cv::CAP_PROP_FOURCC);
  uint32_t cc = static_cast<uint32_t>(fourcc);
  native_nv12_ = (cc == cv::VideoWriter::fourcc('N', 'V', '1', '2'));
  if (native_nv12_) {
    cap_.set(cv::CAP_PROP_FORMAT, -1);  // raw NV12, no BGR auto-convert
  }

  // Create a VB pool for frame buffers.
  uint32_t aw = ALIGN(static_cast<uint32_t>(width_), DEFAULT_ALIGN);
  uint32_t ah = ALIGN(static_cast<uint32_t>(height_), DEFAULT_ALIGN);
  vb_blk_size_ = COMMON_GetPicBufferSize(aw, ah, PIXEL_FORMAT_NV21,
                                          DATA_BITWIDTH_8, COMPRESS_MODE_NONE,
                                          DEFAULT_ALIGN);
  VB_POOL_CONFIG_S pool_cfg;
  memset(&pool_cfg, 0, sizeof(pool_cfg));
  pool_cfg.u32BlkSize = vb_blk_size_;
  pool_cfg.u32BlkCnt  = 6;  // prefetch(2) + main + infer + venc + spare
  pool_cfg.enRemapMode = VB_REMAP_MODE_CACHED;
  snprintf(pool_cfg.acName, sizeof(pool_cfg.acName), "usb_cam_%d", device);

  vb_pool_ = CVI_VB_CreatePool(&pool_cfg);
  if (vb_pool_ == VB_INVALID_POOLID) {
    cap_.release();
    throw std::runtime_error("UsbCamera: failed to create VB pool (" +
                             std::to_string(vb_blk_size_) + " x 3)");
  }

  const char* fmt_name = native_nv12_ ? "NV12(fast)" : "BGR";
  char fourcc_str[5] = {};
  fourcc_str[0] = cc & 0xFF;
  fourcc_str[1] = (cc >> 8) & 0xFF;
  fourcc_str[2] = (cc >> 16) & 0xFF;
  fourcc_str[3] = (cc >> 24) & 0xFF;
  LOGI("[UsbCamera] opened /dev/video%d  %dx%d  fourcc=%s  fmt=%s  "
       "vb_pool=%u blk_size=%u  prefetch=on\n",
       device, actual_w, actual_h, fourcc_str, fmt_name,
       (unsigned)vb_pool_, vb_blk_size_);

  // Start prefetch thread — it immediately begins capturing the first frame
  // so it may already be ready by the time Python calls read().
  //
  // Set to low priority (SCHED_IDLE) so that BGR→NV21 color conversion
  // does not compete with inference's CPU steps (VPSS setup, NMS) on
  // single-core SoCs like CV181x.  The prefetch thread yields CPU time
  // to inference whenever the scheduler has a choice, but still runs
  // during hardware waits (VPSS/TPU/VENC).
  capture_thread_ = std::thread(&PyUsbCamera::captureLoop, this);
  {
    struct sched_param sp = {};
    sp.sched_priority = 0;
    pthread_setschedparam(capture_thread_.native_handle(), SCHED_IDLE, &sp);
  }
}

PyUsbCamera::~PyUsbCamera() { close(); }

// ---------------------------------------------------------------------------
// captureOne — capture a single frame into VB pool memory.
//
// This runs on the prefetch thread.  It allocates a VB block, maps it,
// reads from the camera, converts to NV21, sets PTS, and flushes the cache.
// Returns nullptr if the camera fails to deliver a frame.
// ---------------------------------------------------------------------------
std::unique_ptr<PyUsbCamera::PrefetchedFrame>
PyUsbCamera::captureOne() {
  const uint32_t w = static_cast<uint32_t>(width_);
  const uint32_t h = static_cast<uint32_t>(height_);

  // ── Get a VB block from our pool ────────────────────────────────────────
  // Retry with back-off if the pool is temporarily exhausted (all blocks
  // held by Python / encoder).  Give up after ~500ms.
  VB_BLK blk = VB_INVALID_HANDLE;
  for (int attempt = 0; attempt < 50 && !stop_thread_; ++attempt) {
    blk = CVI_VB_GetBlock(vb_pool_, vb_blk_size_);
    if (blk != VB_INVALID_HANDLE) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (blk == VB_INVALID_HANDLE) {
    return nullptr;
  }

  CVI_U64 phy_addr = CVI_VB_Handle2PhysAddr(blk);
  VB_POOL pool_id  = CVI_VB_Handle2PoolId(blk);

  void* vir_base = CVI_SYS_MmapCache(phy_addr, vb_blk_size_);
  if (!vir_base) {
    CVI_VB_ReleaseBlock(blk);
    return nullptr;
  }

  // ── Build VIDEO_FRAME_INFO_S ────────────────────────────────────────────
  VB_CAL_CONFIG_S vb_cal;
  COMMON_GetPicBufferConfig(w, h, PIXEL_FORMAT_NV21, DATA_BITWIDTH_8,
                            COMPRESS_MODE_NONE, DEFAULT_ALIGN, &vb_cal);

  VIDEO_FRAME_INFO_S frame_info;
  memset(&frame_info, 0, sizeof(frame_info));
  frame_info.u32PoolId = pool_id;

  VIDEO_FRAME_S* vf = &frame_info.stVFrame;
  vf->enCompressMode = COMPRESS_MODE_NONE;
  vf->enPixelFormat  = PIXEL_FORMAT_NV21;
  vf->enVideoFormat  = VIDEO_FORMAT_LINEAR;
  vf->enColorGamut   = COLOR_GAMUT_BT709;
  vf->enDynamicRange = DYNAMIC_RANGE_SDR8;
  vf->u32Width       = w;
  vf->u32Height      = h;
  vf->u32Stride[0]   = vb_cal.u32MainStride;
  vf->u32Stride[1]   = vb_cal.u32CStride;
  vf->u32Stride[2]   = vb_cal.u32CStride;
  vf->u32Length[0]   = vb_cal.u32MainYSize;
  vf->u32Length[1]   = vb_cal.u32MainCSize;

  vf->u64PhyAddr[0]  = phy_addr;
  vf->u64PhyAddr[1]  = phy_addr +
      ALIGN(vb_cal.u32MainYSize, vb_cal.u16AddrAlign);
  vf->pu8VirAddr[0]  = static_cast<uint8_t*>(vir_base);
  vf->pu8VirAddr[1]  = static_cast<uint8_t*>(vir_base) +
      ALIGN(vb_cal.u32MainYSize, vb_cal.u16AddrAlign);

  uint8_t* dst_y  = vf->pu8VirAddr[0];
  uint8_t* dst_vu = vf->pu8VirAddr[1];
  const uint32_t y_stride  = vf->u32Stride[0];
  const uint32_t uv_stride = vf->u32Stride[1];

  // ── Capture and convert to NV21 ─────────────────────────────────────────
  cv::Mat raw;
  if (!cap_.read(raw) || raw.empty()) {
    CVI_SYS_Munmap(vir_base, vb_blk_size_);
    CVI_VB_ReleaseBlock(blk);
    return nullptr;
  }

  if (native_nv12_) {
    // Fast path: NV12 -> NV21 (swap U<->V)
    for (uint32_t row = 0; row < h; ++row)
      std::memcpy(dst_y + row * y_stride, raw.data + row * w, w);

    const uint8_t* uv_src = raw.data + static_cast<size_t>(h) * w;
    const uint32_t half_h = h / 2;
    const uint32_t half_w = w / 2;
    for (uint32_t row = 0; row < half_h; ++row) {
      const uint8_t* s = uv_src + row * w;
      uint8_t*       d = dst_vu + row * uv_stride;
      for (uint32_t col = 0; col < half_w; ++col) {
        d[col * 2]     = s[col * 2 + 1];  // V
        d[col * 2 + 1] = s[col * 2];      // U
      }
    }

  } else {
    // Fallback path: BGR -> I420 -> NV21
    cv::Mat& bgr = raw;
    if (static_cast<uint32_t>(bgr.cols) != w ||
        static_cast<uint32_t>(bgr.rows) != h) {
      cv::resize(bgr, bgr, cv::Size(static_cast<int>(w), static_cast<int>(h)));
    }

    cv::Mat i420;
    cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

    const uint8_t* y_src = i420.data;
    for (uint32_t row = 0; row < h; ++row)
      std::memcpy(dst_y + row * y_stride, y_src + row * w, w);

    const uint32_t half_w = w / 2;
    const uint32_t half_h = h / 2;
    const uint8_t* u_src = i420.data + static_cast<size_t>(h) * w;
    const uint8_t* v_src = i420.data + static_cast<size_t>(h) * w * 5 / 4;
    for (uint32_t row = 0; row < half_h; ++row) {
      uint8_t*       d = dst_vu + row * uv_stride;
      const uint8_t* u = u_src + row * half_w;
      const uint8_t* v = v_src + row * half_w;
      for (uint32_t col = 0; col < half_w; ++col) {
        d[col * 2]     = v[col];
        d[col * 2 + 1] = u[col];
      }
    }
  }

  // ── PTS: monotonically increasing microseconds ──────────────────────────
  {
    auto now = std::chrono::steady_clock::now();
    uint64_t pts_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            now - pts_origin_).count());
    vf->u64PTS = pts_us;
  }

  // Flush CPU cache so hardware encoder sees the data.
  CVI_SYS_IonFlushCache(phy_addr, vir_base, vb_blk_size_);

  auto pf = std::unique_ptr<PrefetchedFrame>(new PrefetchedFrame());
  pf->frame_info  = frame_info;
  pf->blk         = blk;
  pf->vir_base    = vir_base;
  pf->mapped_size = vb_blk_size_;
  return pf;
}

// ---------------------------------------------------------------------------
// captureLoop — background thread that prefetches the next frame.
//
// Pipeline: while Python processes frame N (inference + encode), this thread
// captures frame N+1.  When read() is called, the frame is already waiting.
// ---------------------------------------------------------------------------
void PyUsbCamera::captureLoop() {
  while (true) {
    // Capture a frame (blocking on V4L2 read).
    auto frame = captureOne();

    std::unique_lock<std::mutex> lk(prefetch_mutex_);

    if (stop_thread_) {
      // Release the frame we just captured (if any) and exit.
      if (frame) releasePrefetched(frame.get());
      return;
    }

    if (!frame) {
      capture_error_ = "UsbCamera: camera stopped delivering frames";
      prefetch_cv_.notify_one();
      return;
    }

    // Wait until the consumer (read()) takes the previous frame.
    // This ensures we hold at most 1 prefetched VB block.
    prefetch_cv_.wait(lk, [this] {
      return !prefetched_ || stop_thread_;
    });

    if (stop_thread_) {
      releasePrefetched(frame.get());
      return;
    }

    // Store the new frame and wake up read().
    prefetched_ = std::move(frame);
    prefetch_cv_.notify_one();
  }
}

// ---------------------------------------------------------------------------
// read — return the next frame from the prefetch thread.
// ---------------------------------------------------------------------------
PyImage PyUsbCamera::read() {
  if (closed_) throw std::runtime_error("UsbCamera: device is closed");

  std::unique_ptr<PrefetchedFrame> pf;

  {
    // Release GIL while waiting for the prefetch thread — the wait can
    // block for 100+ ms and holding the GIL would stall the inference
    // thread (which needs the GIL to return results).
    py::gil_scoped_release nogil;

    std::unique_lock<std::mutex> lk(prefetch_mutex_);

    // Wait until a frame is ready or an error occurred.
    prefetch_cv_.wait(lk, [this] {
      return prefetched_ != nullptr || !capture_error_.empty() || stop_thread_;
    });

    if (!capture_error_.empty()) {
      throw std::runtime_error(capture_error_);
    }
    if (stop_thread_ || !prefetched_) {
      throw std::runtime_error("UsbCamera: device is closed");
    }

    pf = std::move(prefetched_);

    // Wake the capture thread to start fetching the next frame.
    prefetch_cv_.notify_one();
  }

  // Wrap in VbBackedImage (releases VB block when Python drops the reference).
  auto img = std::make_shared<VbBackedImage>(
      pf->frame_info, pf->blk, pf->vir_base, pf->mapped_size);

  // Prevent the PrefetchedFrame destructor from releasing resources —
  // ownership transferred to VbBackedImage.
  pf->vir_base = nullptr;

  std::shared_ptr<BaseImage> base = img;
  return PyImage(base);
}

void PyUsbCamera::close() {
  if (!closed_) {
    // Signal the capture thread to stop.
    {
      std::lock_guard<std::mutex> lk(prefetch_mutex_);
      stop_thread_ = true;
    }
    prefetch_cv_.notify_all();

    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }

    // Release any unconsumed prefetched frame.
    if (prefetched_) {
      releasePrefetched(prefetched_.get());
      prefetched_.reset();
    }

    // Now safe to release camera and VB pool.
    cap_.release();
    if (vb_pool_ != VB_INVALID_POOLID) {
      CVI_VB_DestroyPool(vb_pool_);
      vb_pool_ = VB_INVALID_POOLID;
    }
    closed_ = true;
  }
}

bool PyUsbCamera::isOpened() const {
  return !closed_ && cap_.isOpened();
}

}  // namespace pytdl
