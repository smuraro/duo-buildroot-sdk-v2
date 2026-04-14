#ifndef PYTHON_RTSP_CLIENT_HPP_
#define PYTHON_RTSP_CLIENT_HPP_

#include <pybind11/pybind11.h>
#include <atomic>
#include <memory>
#include <string>
#include "opencv2/opencv.hpp"
#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

// RTSP / video-file client that decodes frames and wraps them in a VPSSImage
// so the output is compatible with inference (VPSS preprocessor) and with
// RTSPServer.send_frame (for re-streaming with overlays).
//
// Decoding is done by OpenCV VideoCapture (backed by FFmpeg), which supports:
//   rtsp://...          RTSP streams (H264/H265)
//   rtmp://...          RTMP streams
//   http://.../.m3u8    HLS
//   /path/to/video.mp4  Local video files
//
// Usage:
//   client = image.RtspClient("rtsp://192.168.1.10:554/live")
//   frame  = client.read()
//   result = model.inference(frame)
//   client.release()      # return frame buffer
//   client.close()        # stop stream
class PyRtspClient {
 public:
  // url:         RTSP/video URL or file path
  // width/height: resize decoded frames to this size (0 = native resolution)
  // timeout_ms:  connection/read timeout in milliseconds (default 5000)
  // transport:   "tcp" (default, more reliable) or "udp"
  explicit PyRtspClient(const std::string& url,
                        int width = 0, int height = 0,
                        int timeout_ms = 5000,
                        const std::string& transport = "tcp");
  ~PyRtspClient();

  // Decode the next frame and return it as a PyImage (VPSSImage with BGR data
  // in ION memory).  Raises RuntimeError on failure or end-of-stream.
  PyImage read();

  // No-op for API compatibility with Camera (VPSSImage manages its own memory).
  void release() {}

  // Close the stream and release decoder resources.
  void close();

  bool isOpened() const;

  // Context manager support.
  PyRtspClient* enter() { return this; }
  void exit(py::object, py::object, py::object) { close(); }

 private:
  cv::VideoCapture cap_;
  int target_width_;
  int target_height_;
  bool closed_ = false;
};

}  // namespace pytdl
#endif  // PYTHON_RTSP_CLIENT_HPP_
