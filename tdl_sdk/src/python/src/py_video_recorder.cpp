#include "py_video_recorder.hpp"
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

PyVideoRecorder::~PyVideoRecorder() { rec_.reset(); }

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
  if (ret != 0) LOGE("[PyVideoRecorder] sendFrame failed: %d", ret);
}

void PyVideoRecorder::rotate() {
  if (rec_) rec_->rotate();
}

void PyVideoRecorder::close() {
  // Explicitly release the recorder to finalize the last MP4 segment.
  // Safe to call multiple times.
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
