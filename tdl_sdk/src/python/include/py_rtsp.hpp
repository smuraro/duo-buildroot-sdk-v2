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
  // bitrate: encoding bitrate in kbps (default 3072).  Higher = better quality
  //          during motion but more bandwidth. 1024–8192 kbps typical range.
  // gop: keyframe interval in frames (default 15). Smaller = sharper during
  //      motion, larger = better compression for static scenes.
  PyRTSP(int32_t width, int32_t height, int32_t chn = 0,
         const std::string& codec = "h264",
         const std::string& session_name = "",
         int32_t bitrate = 3072, int32_t gop = 15, int32_t fps = 25);
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

// Draw a classification / attribute result (CLASSIFICATION, CLS_ATTRIBUTE).
// result: list or dict returned by model.inference().
// Renders a filled label box in the top-left corner.
void drawClassification(PyImage& image, const py::object& result);

// Draw a semantic segmentation overlay (SEGMENTATION).
// result: list/dict with keys "output_width", "output_height", "class_id".
// alpha: blend 0.0 (invisible) … 1.0 (opaque), default 0.5.
void drawSegmentation(PyImage& image, const py::object& result, float alpha);

// Draw instance segmentation: bounding boxes + per-instance mask overlays
// (OBJECT_DETECTION_WITH_SEGMENTATION).
// result: list/dict with keys "mask_width", "mask_height", "bboxes_seg".
// alpha: mask blend factor, default 0.45.
void drawInstanceSegmentation(PyImage& image, const py::object& result,
                               float score_threshold, float alpha);

// Draw OCR text result at the bottom of the frame (OCR_INFO).
// result: list containing a string, as returned by OCR models.
void drawOcr(PyImage& image, const py::object& result);

// Debug: draw a thumbnail of the face crop region in the bottom-right corner.
// face_x1..face_y2: detected face bbox.  thumb_size: thumbnail px (default 96).
// pad_ratio: padding around the face (default 0.2 = 20%).
void drawCropOverlay(PyImage& image, float face_x1, float face_y1,
                     float face_x2, float face_y2,
                     int thumb_size = 96, float pad_ratio = 0.2f);

// Two-phase thumbnail API.
//
// captureFaceCrop reads the pristine face region from the frame NOW (before
// any bbox/label drawing contaminates those pixels) and returns an opaque
// snapshot dict carrying the downsampled Y+UV thumbnail pixels.  The caller
// then draws bounding boxes, classification labels etc., and finally calls
// drawFaceThumbnail to paint the bottom-right preview over the top of those
// drawings — so the preview shows the exact image the stage-2 model received,
// with zero bbox/label bleed into the thumbnail rectangle and zero bbox-edge
// contamination inside the thumbnail content.
//
// Returns an empty dict (py::dict()) when the crop would be invalid (face
// bbox empty, thumbnail too small, or off-screen).  drawFaceThumbnail treats
// an empty dict as a no-op so callers can pass it unconditionally.
py::dict captureFaceCrop(PyImage& image, float face_x1, float face_y1,
                         float face_x2, float face_y2,
                         int thumb_size = 96, float pad_ratio = 0.2f);

void drawFaceThumbnail(PyImage& image, const py::dict& snapshot);

// Return the worst-case bounding rect (x1, y1, x2, y2) of the thumbnail in
// frame pixel coordinates — including the white border. Use this to filter
// face detections that fall inside the thumbnail region (those are the
// detector latching onto the thumbnail painted on the previous frame).
py::tuple getThumbnailRect(PyImage& image, int thumb_size = 96);

}  // namespace pytdl
#endif  // PYTHON_RTSP_HPP_
