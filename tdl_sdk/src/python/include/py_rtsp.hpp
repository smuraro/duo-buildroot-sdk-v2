#ifndef PYTHON_RTSP_HPP_
#define PYTHON_RTSP_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include "encoder/rtsp/rtsp.hpp"
#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

// RTSP streaming server wrapping the C++ RTSP class (VENC + cvi_rtsp/live555).
// Accepts hardware camera frames (VPSSImage) and encodes them in H.264/H.265.
//
// URL: rtsp://<device-ip>:554/<session_name>
//
// Usage:
//   rtsp = tdl.image.RTSPServer(width=1280, height=720)
//   frame = cam.read()
//   tdl.image.draw_detections(frame, detections)
//   rtsp.send_frame(frame)
class PyRTSP {
 public:
  // codec: "h264" (default) or "h265"
  // session_name: URL path component, default = codec name ("h264"/"h265")
  PyRTSP(int32_t width, int32_t height, int32_t chn = 0,
         const std::string& codec = "h264",
         const std::string& session_name = "");
  ~PyRTSP();

  // Send a hardware camera frame (must be a VPSSImage from Camera.read()).
  // After draw_* calls the frame is still valid to send.
  void sendFrame(const PyImage& image);

  // URL path hint: "rtsp://<device-ip>:554/<session_name>"
  std::string getSessionName() const { return session_name_; }

  PyRTSP* enter() { return this; }
  void exit(py::object, py::object, py::object) {}

 private:
  std::unique_ptr<RTSP> rtsp_;
  std::string session_name_;
};

// ─── Draw utilities ──────────────────────────────────────────────────────────
// All functions operate in-place on a hardware camera frame (VPSSImage).
// Internally they convert YUV→BGR, draw with OpenCV, then convert BGR→YUV
// back into the hardware buffer and flush the cache.
// They raise RuntimeError if the image is not a hardware (VPSSImage) frame.

// Draw a single bounding box.  color = (R, G, B) tuple.
void drawBbox(PyImage& image, int x1, int y1, int x2, int y2,
              py::tuple color, int thickness);

// Draw a text string.  scale is the OpenCV font scale.
void drawText(PyImage& image, const std::string& text, int x, int y,
              py::tuple color, double scale);

// Draw all detections (bounding boxes + class label + score) from
// model.inference() output. detections is a list of dicts with keys:
//   "x1","y1","x2","y2","score","class_name"
void drawDetections(PyImage& image, const py::list& detections,
                    float score_threshold);

// Convert a hardware camera frame to JPEG and return Python bytes.
// quality: 0-100 (default 80).  scale: 0<scale<1 downscales before encode
// (e.g. 0.5 → half width/height = 4× fewer pixels → much faster encode).
// Caller must not have released the frame yet.
py::bytes frameToJpeg(const PyImage& image, int quality, float scale);

// Draw keypoints and skeleton lines.  Each entry in detections_with_landmarks
// is a dict as returned by keypoint models with keys:
//   "x1","y1","x2","y2","score",
//   "landmarks_x" (list), "landmarks_y" (list), "landmarks_score" (list)
// COCO-17 skeleton connectivity is used when 17 keypoints are present.
void drawKeypoints(PyImage& image, const py::list& detections_with_landmarks,
                   float score_threshold);

}  // namespace pytdl
#endif  // PYTHON_RTSP_HPP_
