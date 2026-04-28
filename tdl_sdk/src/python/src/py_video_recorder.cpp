#include "py_video_recorder.hpp"
#include <cstdio>
#include "image/base_image.hpp"
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"

namespace pytdl {

static VPSSImage* requireVPSS(const PyImage& image, const char* fn) {
  VPSSImage* vpss = dynamic_cast<VPSSImage*>(image.getImage().get());
  if (!vpss)
    throw std::runtime_error(
        std::string(fn) +
        ": requires a hardware camera frame (VPSSImage). "
        "Use image.Camera.read() to get one.");
  return vpss;
}

PyVideoRecorder::PyVideoRecorder(int32_t width, int32_t height,
                                 const std::string& out_dir,
                                 const std::string& codec,
                                 int32_t segment_seconds, int32_t fps,
                                 int32_t bitrate, int32_t gop,
                                 int32_t chn, int32_t jpeg_chn) {
  rec_ = std::make_unique<VideoRecorder>(
      width, height, out_dir, codec, segment_seconds, fps, bitrate, gop, chn,
      jpeg_chn);
}

PyVideoRecorder::~PyVideoRecorder() {
  if (broken_) {
    // VENC channel is corrupted — releasing it would block in DestroyChn.
    // Leak intentionally; process is exiting anyway.
    (void)rec_.release();
    return;
  }
  rec_.reset();
}

void PyVideoRecorder::sendFrame(const PyImage& image) {
  if (!rec_) throw std::runtime_error("VideoRecorder is closed");
  VPSSImage* vpss = requireVPSS(image, "VideoRecorder.send_frame");
  vpss->flushCache();
  VIDEO_FRAME_INFO_S* fi = vpss->getFrame();
  if (!fi) throw std::runtime_error("VideoRecorder.send_frame: null frame");
  int ret;
  {
    // VENC encode + MP4 write can block for tens of ms — release the GIL.
    py::gil_scoped_release nogil;
    ret = rec_->sendFrame(fi);
  }
  if (ret != 0) {
    // Mark broken so close()/destructor skip rec_.reset() — calling
    // DestroyChn on a VENC channel that returned BUSY (0xC0078012)
    // blocks in the driver waitqueue (D-state, only reboot recovers).
    broken_ = true;
    LOGE("[PyVideoRecorder] sendFrame failed: %d", ret);
    throw std::runtime_error(
        "VideoRecorder.send_frame failed: 0x" +
        [&]{ char b[16]; std::snprintf(b, sizeof(b), "%X", ret); return std::string(b); }() +
        " — recorder unusable, do not call close()");
  }
}

void PyVideoRecorder::rotate() {
  if (rec_) rec_->rotate();
}

void PyVideoRecorder::close() {
  // Explicitly release the recorder to finalize the last MP4 segment.
  // Safe to call multiple times.
  if (broken_) {
    // See ~PyVideoRecorder() — leak instead of triggering D-state.
    (void)rec_.release();
    return;
  }
  // VideoRecorder destructor → VENC StopRecvFrame/DestroyChn can block
  // for hundreds of ms (waiting encoder drain). Release the GIL so the
  // Python main thread (or watchdog) can run during this time.
  py::gil_scoped_release nogil;
  rec_.reset();
}

std::string PyVideoRecorder::currentSegment() const {
  return rec_ ? rec_->currentSegment() : std::string();
}

std::string PyVideoRecorder::outputDir() const {
  return rec_ ? rec_->outputDir() : std::string();
}

int64_t PyVideoRecorder::segmentStartMs() const {
  return rec_ ? rec_->segmentStartMs() : 0;
}

py::list PyVideoRecorder::history() const {
  py::list out;
  if (!rec_) return out;
  for (const auto& s : rec_->history()) {
    py::dict d;
    d["filename"]    = s.filename;
    d["thumbnail"]   = s.thumbnail;
    d["size_bytes"]  = s.size_bytes;
    d["started_ms"]  = s.started_ms;
    d["duration_ms"] = s.duration_ms;
    d["frame_count"] = s.frame_count;
    out.append(d);
  }
  return out;
}

}  // namespace pytdl
