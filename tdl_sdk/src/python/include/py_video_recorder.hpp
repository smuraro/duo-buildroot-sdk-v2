#ifndef PYTHON_VIDEO_RECORDER_HPP_
#define PYTHON_VIDEO_RECORDER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include "encoder/video_recorder/video_recorder.hpp"
#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

// Hardware-encoded DVR recorder: captures VPSSImage frames, encodes with VENC
// (H.264 or H.265), and writes time-rotated MP4 segments.  On each segment
// start a JPEG thumbnail is also saved next to the video file.
//
// Usage:
//   rec = tdl.image.VideoRecorder(1280, 720, "/mnt/sd/dvr",
//                                 codec="h264", segment_seconds=30, fps=15)
//   frame = cam.read()
//   rec.send_frame(frame)        # encodes + writes
//   cam.release()
//   ...
//   rec.close()                  # finalize last segment
class PyVideoRecorder {
 public:
  PyVideoRecorder(int32_t width, int32_t height,
                  const std::string& out_dir,
                  const std::string& codec = "h264",
                  int32_t segment_seconds = 30,
                  int32_t fps = 15,
                  int32_t bitrate = 3072,
                  int32_t gop = 15,
                  int32_t chn = 0,
                  int32_t jpeg_chn = 1);
  ~PyVideoRecorder();

  void sendFrame(const PyImage& image);
  void rotate();
  void close();

  std::string currentSegment() const;
  std::string outputDir() const;
  int64_t     segmentStartMs() const;
  py::list    history() const;     // list of dicts (one per closed segment)

  PyVideoRecorder* enter() { return this; }
  void exit(py::object, py::object, py::object) { close(); }

 private:
  std::unique_ptr<VideoRecorder> rec_;
};

}  // namespace pytdl
#endif  // PYTHON_VIDEO_RECORDER_HPP_
