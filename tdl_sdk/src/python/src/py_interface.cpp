#include "py_camera.hpp"
#include "py_image.hpp"
//#include "py_llm.hpp"
#include "py_matcher.hpp"
#include "py_model.hpp"
#include "py_rtsp.hpp"
#include "py_video_recorder.hpp"
#ifdef HAVE_OPENCV_VIDEOIO
#include "py_rtsp_client.hpp"
#include "py_usb_camera.hpp"
#endif
#include "py_rtsp_client_vdec.hpp"
#include "utils/tokenizer_bpe.hpp"
#include "nn/tdl_model_factory.hpp"
#include "nn/tdl_model_defs.hpp"
using namespace pytdl;
using namespace pybind11::literals;

// 明确指定函数类型，解决重载问题
PyImage (*read_func)(const std::string&) = &read;
void (*write_func)(const PyImage&, const std::string&) = &write;
PyImage (*resize_func)(const PyImage&, int, int) = &resize;
PyImage (*crop_func)(const PyImage&,
                     const std::tuple<int, int, int, int>&) = &crop;
PyImage (*crop_resize_func)(const PyImage&,
                            const std::tuple<int, int, int, int>&, int,
                            int) = &cropResize;
PyImage (*align_face_func)(const PyImage& image,
                           const std::vector<float>& src_landmark_xy,
                           const std::vector<float>& dst_landmark_xy,
                           int num_points) = &align_face;

// 使用不同的名字区分两个get_model函数
PyModel (*get_model_with_path)(ModelType, const std::string&, const py::dict&,
                               const int) = &get_model;
PyModel (*get_model_with_dir)(ModelType, const std::string&,
                              const int) = &get_model_from_dir;


// pybind11绑定实现
PYBIND11_MODULE(tdl, m) {
  m.doc() = "tdl sdk module python binding";

  // 图像模块
  py::module image = m.def_submodule("image", "image module");
  // 绑定枚举
  py::enum_<ImageFormat>(image, "ImageFormat")
      .value("RGB_PLANAR", ImageFormat::RGB_PLANAR)
      .value("BGR_PLANAR", ImageFormat::BGR_PLANAR)
      .value("RGB_PACKED", ImageFormat::RGB_PACKED)
      .value("BGR_PACKED", ImageFormat::BGR_PACKED)
      .value("GRAY", ImageFormat::GRAY)
      .value("YUV420SP_UV", ImageFormat::YUV420SP_UV)
      .value("YUV420SP_VU", ImageFormat::YUV420SP_VU)
      //   .value("NV12", ImageFormat::YUV420SP_UV)
      //   .value("NV21", ImageFormat::YUV420SP_VU)
      .value("YUV420P_UV", ImageFormat::YUV420P_UV)
      .value("YUV420P_VU", ImageFormat::YUV420P_VU)
      .value("YUV422P_UV", ImageFormat::YUV422P_UV)
      .value("YUV422P_VU", ImageFormat::YUV422P_VU)
      .value("YUV422SP_UV", ImageFormat::YUV422SP_UV)
      .value("YUV422SP_VU", ImageFormat::YUV422SP_VU)
      .export_values();

  py::enum_<TDLDataType>(image, "TDLDataType")
      .value("UINT8", TDLDataType::UINT8)
      .value("INT8", TDLDataType::INT8)
      .value("UINT16", TDLDataType::UINT16)
      .value("INT16", TDLDataType::INT16)
      .value("UINT32", TDLDataType::UINT32)
      .value("INT32", TDLDataType::INT32)
      .value("FP32", TDLDataType::FP32)
      .export_values();

  // 绑定图像类
  py::class_<PyImage>(image, "Image")
      .def(py::init<>())
      .def_static("from_numpy", &PyImage::fromNumpy, py::arg("numpy_array"),
                  py::arg("format") = ImageFormat::BGR_PACKED)
      .def("get_size", &PyImage::getSize)
      .def("get_format", &PyImage::getFormat);

  // 绑定模块函数，使用明确类型的函数指针
  image.def("write", write_func, py::arg("image"), py::arg("path"));
  image.def("read", read_func, py::arg("path"));
  image.def("resize", resize_func, py::arg("src"), py::arg("width"),
            py::arg("height"));
  image.def("crop", crop_func, py::arg("src"), py::arg("roi"));
  image.def("crop_resize", crop_resize_func, py::arg("src"), py::arg("roi"),
            py::arg("width"), py::arg("height"));
  image.def("align_face", align_face_func, py::arg("image"),
            py::arg("src_landmark_xy"), py::arg("dst_landmark_xy"),
            py::arg("num_points"));
  image.def(
      "from_numpy",
      [](const py::array& arr, ImageFormat format) {
        return PyImage(arr, format);
      },
      py::arg("numpy_array"), py::arg("format") = ImageFormat::BGR_PACKED);

  // 摄像头捕获类
  py::class_<PyCamera>(image, "Camera")
      .def(py::init<int32_t, int32_t, ImageFormat, int32_t, bool, bool>(),
           py::arg("width"), py::arg("height"),
           py::arg("format") = ImageFormat::YUV420SP_VU,
           py::arg("vb_buffer_num") = 3,
           py::arg("mirror") = false,
           py::arg("flip") = false,
           "Open the camera.\n"
           "mirror: horizontal flip (left ↔ right), done in VPSS hardware.\n"
           "flip:  vertical flip   (top  ↔ bottom), done in VPSS hardware.")
      .def("read", &PyCamera::read, py::arg("channel") = 0,
           "Capture one frame from the camera and return it as an Image")
      .def("release", &PyCamera::release, py::arg("channel") = 0,
           "Release the frame buffer back to the VB pool")
      .def("close", &PyCamera::close, "Stop the camera and free resources")
      .def("__enter__", &PyCamera::enter, py::return_value_policy::reference)
      .def("__exit__", &PyCamera::exit);

#ifdef HAVE_OPENCV_VIDEOIO
  // RTSP / video-file client (software decode via OpenCV/FFmpeg)
  py::class_<PyRtspClient>(image, "RtspClient")
      .def(py::init<const std::string&, int, int, int, const std::string&>(),
           py::arg("url"),
           py::arg("width") = 0, py::arg("height") = 0,
           py::arg("timeout_ms") = 5000,
           py::arg("transport") = "tcp",
           "Open an RTSP stream or video file for decoding (software decode).\n"
           "url:        RTSP/RTMP/HLS URL or local video file path.\n"
           "width/height: resize frames to this resolution (0 = native).\n"
           "timeout_ms: connection and read timeout in milliseconds.\n"
           "transport:  'tcp' (default, reliable) or 'udp' (lower latency).")
      .def("read", &PyRtspClient::read,
           "Decode the next frame and return it as a VPSSImage.\n"
           "Compatible with model.inference() and RTSPServer.send_frame().\n"
           "Raises RuntimeError on end-of-stream or timeout.")
      .def("release", &PyRtspClient::release,
           "No-op. Provided for API compatibility with Camera.")
      .def("close", &PyRtspClient::close, "Close the stream and free decoder resources.")
      .def("is_opened", &PyRtspClient::isOpened, "Return True if the stream is open.")
      .def("__enter__", &PyRtspClient::enter, py::return_value_policy::reference)
      .def("__exit__", &PyRtspClient::exit);
#endif  // HAVE_OPENCV_VIDEOIO

#ifdef HAVE_OPENCV_VIDEOIO
  // USB camera capture via V4L2 (OpenCV VideoCapture backend)
  py::class_<PyUsbCamera>(image, "UsbCamera")
      .def(py::init<int, int, int>(),
           py::arg("device") = 0,
           py::arg("width")  = 640,
           py::arg("height") = 480,
           "Open a USB (UVC) camera via V4L2.\n"
           "device: V4L2 device index (0 = /dev/video0, 1 = /dev/video1, ...).\n"
           "width / height: requested capture resolution.\n"
           "Frames are returned as VPSSImage (NV12) compatible with\n"
           "model.inference(), RTSPServer.send_frame(), and all draw_* functions.\n"
           "If the camera natively outputs NV12, no color conversion is performed.")
      .def("read", &PyUsbCamera::read,
           "Capture the next frame and return it as a VPSSImage (NV12).\n"
           "Compatible with model.inference() and RTSPServer.send_frame().\n"
           "Raises RuntimeError on failure.")
      .def("release", &PyUsbCamera::release,
           "No-op. Provided for API compatibility with Camera.")
      .def("close", &PyUsbCamera::close,
           "Close the device and free resources.")
      .def_property_readonly("width", &PyUsbCamera::getWidth,
           "Actual capture width in pixels")
      .def_property_readonly("height", &PyUsbCamera::getHeight,
           "Actual capture height in pixels")
      .def("is_opened", &PyUsbCamera::isOpened,
           "Return True if the device is open.")
      .def("__enter__", &PyUsbCamera::enter, py::return_value_policy::reference)
      .def("__exit__", &PyUsbCamera::exit);
#endif  // HAVE_OPENCV_VIDEOIO

  // Hardware-accelerated RTSP client (live555 + VDEC)
  py::class_<PyRtspClientVdec>(image, "RtspClientVdec")
      .def(py::init<const std::string&, int, int, int, const std::string&>(),
           py::arg("url"),
           py::arg("width") = 0, py::arg("height") = 0,
           py::arg("timeout_ms") = 5000,
           py::arg("transport") = "tcp",
           "Open an RTSP stream using live555 (RTSP/RTP) + VDEC hardware decode.\n"
           "H264 and H265 streams are decoded by the VDEC unit — zero CPU cost.\n"
           "Decoded frames are YUV420 NV12 in VB memory, compatible with inference.\n"
           "url:         RTSP stream URL (rtsp://...).\n"
           "width/height: maximum decode resolution; 0 = use native stream resolution.\n"
           "timeout_ms:  per-frame CVI_VDEC_GetFrame timeout in milliseconds.\n"
           "transport:   'tcp' (default, more reliable) or 'udp' (lower latency).")
      .def("read", &PyRtspClientVdec::read,
           "Decode the next frame via VDEC hardware.\n"
           "Returns a VPSSImage (YUV420 NV12) compatible with model.inference().\n"
           "IMPORTANT: call release() before calling read() again.")
      .def("release", &PyRtspClientVdec::release,
           "Return the current frame buffer to the VDEC pool.\n"
           "Must be called after each read() before the next read().")
      .def("pin_for_inference", &PyRtspClientVdec::pinForInference,
           "Move the current held frame to the inference slot so the inference\n"
           "thread can safely read it while read() fetches the next frame.\n"
           "Call AFTER send_frame() (no more writes to the frame).")
      .def("release_inference", &PyRtspClientVdec::releaseInference,
           "Release the inference slot back to the VDEC pool.\n"
           "Call AFTER the inference thread has finished (future.result() returned).")
      .def("close", &PyRtspClientVdec::close,
           "Stop the stream and release all resources.")
      .def("is_opened", &PyRtspClientVdec::isOpened,
           "Return True if the stream is open and VDEC is running.")
      .def("__enter__", &PyRtspClientVdec::enter,
           py::return_value_policy::reference)
      .def("__exit__", &PyRtspClientVdec::exit);

  // RTSP streaming server
  py::class_<PyRTSP>(image, "RTSPServer")
      .def(py::init<int32_t, int32_t, int32_t, const std::string&,
                    const std::string&, int32_t, int32_t, int32_t>(),
           py::arg("width"), py::arg("height"), py::arg("chn") = 0,
           py::arg("codec") = "h264", py::arg("session_name") = "",
           py::arg("bitrate") = 3072, py::arg("gop") = 15, py::arg("fps") = 25,
           "Create an RTSP server.  Access stream at rtsp://<ip>:554/<session_name>.\n"
           "codec: 'h264' (default) or 'h265'.\n"
           "session_name: URL path (defaults to codec name).\n"
           "bitrate: encoding bitrate in kbps (default 3072). Higher = better quality during motion.\n"
           "gop: keyframe interval in frames (default 15). Smaller = sharper during motion.\n"
           "fps: source/destination frame rate (default 25). MUST match the actual frame rate\n"
           "     your application sends frames. Wrong value causes poor quality (rate control\n"
           "     mis-allocation) and regions that do not update visually.")
      .def("send_frame", &PyRTSP::sendFrame, py::arg("frame"),
           "Encode and send a hardware camera frame over RTSP.\n"
           "frame must be a VPSSImage obtained from Camera.read().")
      .def("get_session_name", &PyRTSP::getSessionName,
           "Return the URL path component, e.g. 'h264'.")
      .def("__enter__", &PyRTSP::enter, py::return_value_policy::reference)
      .def("__exit__", &PyRTSP::exit);

  // Hardware DVR recorder: encodes VPSSImage frames with VENC and writes
  // time-rotated MP4 segments (one JPEG thumbnail per segment).
  py::class_<PyVideoRecorder>(image, "VideoRecorder")
      .def(py::init<int32_t, int32_t, const std::string&,
                    const std::string&, int32_t, int32_t, int32_t,
                    int32_t, int32_t, int32_t>(),
           py::arg("width"), py::arg("height"), py::arg("out_dir"),
           py::arg("codec") = "h264",
           py::arg("segment_seconds") = 30,
           py::arg("fps") = 15,
           py::arg("bitrate") = 3072,
           py::arg("gop") = 15,
           py::arg("chn") = 0,
           py::arg("jpeg_chn") = 1,
           "Create a hardware video recorder that writes MP4 segments.\n"
           "out_dir: output directory (created if missing, e.g. '/mnt/sd/dvr').\n"
           "codec: 'h264' or 'h265'.\n"
           "segment_seconds: segment duration before rotating to a new file.\n"
           "fps: must match the rate at which send_frame() is called.\n"
           "bitrate: VENC bitrate in kbps (CBR).\n"
           "gop: keyframe interval in frames (smaller = better seek).\n"
           "chn: VENC channel for video encoding (default 0).\n"
           "jpeg_chn: VENC channel used transiently for JPEG thumbnails "
           "(default 1).  Do not reuse either channel with RTSPServer.")
      .def("send_frame", &PyVideoRecorder::sendFrame, py::arg("frame"),
           "Encode one hardware frame and append to the current segment.\n"
           "Rotates to a new MP4 file once segment_seconds has elapsed "
           "(rotation happens at the next keyframe).")
      .def("rotate", &PyVideoRecorder::rotate,
           "Request the current segment to be closed at the next keyframe.")
      .def("close", &PyVideoRecorder::close,
           "Finalize the current MP4 segment and release VENC resources.")
      .def("current_segment", &PyVideoRecorder::currentSegment,
           "Basename (no directory) of the segment being written, "
           "or '' if no segment is open yet.")
      .def("output_dir", &PyVideoRecorder::outputDir,
           "Output directory as passed to the constructor.")
      .def("segment_start_ms", &PyVideoRecorder::segmentStartMs,
           "Wall-clock epoch time (ms) when the current segment was opened.\n"
           "Returns 0 while no segment is open. Use it to align sidecar\n"
           "files (e.g. detection JSONL) with the MP4 PTS timeline.")
      .def("history", &PyVideoRecorder::history,
           "List of dicts describing all closed segments since start.\n"
           "Keys: filename, thumbnail, size_bytes, started_ms, "
           "duration_ms, frame_count.")
      .def("__enter__", &PyVideoRecorder::enter,
           py::return_value_policy::reference)
      .def("__exit__", &PyVideoRecorder::exit);

  // Draw utilities (operate in-place on hardware camera frames)
  image.def("draw_bbox", &drawBbox,
            py::arg("frame"), py::arg("x1"), py::arg("y1"),
            py::arg("x2"), py::arg("y2"),
            py::arg("color") = py::make_tuple(0, 255, 0),
            py::arg("thickness") = 2,
            "Draw a bounding box on a hardware frame.  color=(R,G,B).");

  image.def("draw_text", &drawText,
            py::arg("frame"), py::arg("text"), py::arg("x"), py::arg("y"),
            py::arg("color") = py::make_tuple(0, 255, 0),
            py::arg("scale") = 0.5,
            "Draw a text string on a hardware frame.  color=(R,G,B).");

  image.def("draw_detections", &drawDetections,
            py::arg("frame"), py::arg("detections"),
            py::arg("score_threshold") = 0.0f,
            "Draw bounding boxes and labels for all detections on a hardware frame.\n"
            "'detections' is the list returned by Model.inference().");

  image.def("draw_keypoints", &drawKeypoints,
            py::arg("frame"), py::arg("detections"),
            py::arg("score_threshold") = 0.0f,
            "Draw keypoints and skeleton lines on a hardware frame.\n"
            "Uses COCO-17 skeleton when 17 keypoints are detected.");

  image.def("draw_classification", &drawClassification,
            py::arg("frame"), py::arg("result"),
            "Draw classification or attribute result (CLASSIFICATION, CLS_ATTRIBUTE).\n"
            "Renders a label box in the top-left corner of the frame.");

  image.def("draw_segmentation", &drawSegmentation,
            py::arg("frame"), py::arg("result"),
            py::arg("alpha") = 0.5f,
            "Draw semantic segmentation overlay (SEGMENTATION).\n"
            "alpha: blend factor 0.0-1.0 (default 0.5).");

  image.def("draw_instance_segmentation", &drawInstanceSegmentation,
            py::arg("frame"), py::arg("result"),
            py::arg("score_threshold") = 0.0f,
            py::arg("alpha") = 0.45f,
            "Draw instance segmentation: bboxes + mask overlays\n"
            "(OBJECT_DETECTION_WITH_SEGMENTATION).\n"
            "alpha: mask blend factor (default 0.45).");

  image.def("draw_ocr", &drawOcr,
            py::arg("frame"), py::arg("result"),
            "Draw OCR text result at the bottom of the frame (OCR_INFO).");

  image.def("draw_crop_overlay", &drawCropOverlay,
            py::arg("frame"), py::arg("x1"), py::arg("y1"),
            py::arg("x2"), py::arg("y2"),
            py::arg("thumb_size") = 96, py::arg("pad_ratio") = 0.2f,
            "Debug: draw a thumbnail of a face crop region in the bottom-right\n"
            "corner of the frame. x1,y1,x2,y2: face bounding box.\n"
            "thumb_size: thumbnail size in px (default 96).\n"
            "pad_ratio: padding around face (default 0.2 = 20%%).");

  image.def("capture_face_crop", &captureFaceCrop,
            py::arg("frame"), py::arg("x1"), py::arg("y1"),
            py::arg("x2"), py::arg("y2"),
            py::arg("thumb_size") = 96, py::arg("pad_ratio") = 0.2f,
            "Capture a pristine snapshot of the face crop region BEFORE any\n"
            "bbox/label drawing. Returns an opaque dict to be passed to\n"
            "draw_face_thumbnail after the drawings are done. Returns an empty\n"
            "dict if the crop is invalid (then draw_face_thumbnail is a no-op).");

  image.def("draw_face_thumbnail", &drawFaceThumbnail,
            py::arg("frame"), py::arg("snapshot"),
            "Paint the face thumbnail captured by capture_face_crop into the\n"
            "bottom-right corner of the frame. Call this AFTER draw_detections\n"
            "and draw_classification so the preview is not contaminated by\n"
            "bbox/label pixels.");

  image.def("get_thumbnail_rect", &getThumbnailRect,
            py::arg("frame"), py::arg("thumb_size") = 96,
            "Return (x1, y1, x2, y2) of the worst-case thumbnail rectangle in\n"
            "frame pixel coordinates, including the white border. Use this to\n"
            "drop face detections whose bbox falls inside the thumbnail region\n"
            "(they are the detector re-detecting the thumbnail painted on the\n"
            "previous frame).");

  image.def("frame_to_jpeg", &frameToJpeg,
            py::arg("frame"), py::arg("quality") = 80, py::arg("scale") = 1.0f,
            "Convert a hardware camera frame (VPSSImage) to JPEG bytes.\n"
            "quality: 0-100 JPEG quality (default 80).\n"
            "scale: downscale factor 0<s<1 before encoding (e.g. 0.5 = half size,\n"
            "  4× fewer pixels, much faster encode). Default 1.0 = full resolution.\n"
            "Call after draw_detections/draw_keypoints, before cam.release().\n"
            "Returns bytes suitable for base64-encoding or HTTP delivery.");

  // 神经网络模块
  py::module nn = m.def_submodule("nn", "Neural network algorithms module");
  py::enum_<ModelType> model_type_enum(nn, "ModelType");
#define X(name, comment) model_type_enum.value(#name, ModelType::name);
  // 直接用 MODEL_TYPE_LIST 把所有 name 都展开一次
  MODEL_TYPE_LIST
#undef X
  model_type_enum.export_values();

  py::class_<PyModel>(nn, "Model")
      .def("close", &PyModel::close,
           "Release the model and its VPSS preprocessor group immediately.\n"
           "Call before script exit to avoid exhausting VPSS groups.")
      .def("__enter__", [](PyModel& m) -> PyModel& { return m; })
      .def("__exit__", [](PyModel& m, py::object, py::object, py::object) { m.close(); })
      .def("get_preprocess_parameters", &PyModel::getPreprocessParameters)
      .def("inference", py::overload_cast<const PyImage&>(&PyModel::inference),
           py::arg("image"))
      .def("inference",
           py::overload_cast<const py::array_t<unsigned char,
                                               py::array::c_style>&>(
               &PyModel::inference),
           py::arg("array"))
      .def("inference",
           py::overload_cast<const PyImage&, const py::dict&>(
               &PyModel::inference),
           py::arg("image"), py::arg("parameters"),
           "Run inference with extra runtime parameters (e.g. score threshold)")
      .def("set_threshold", &PyModel::setThreshold, py::arg("threshold"))
      .def("get_threshold", &PyModel::getThreshold)
      .def("set_soft_nms", &PyModel::setSoftNms,
           py::arg("enable"), py::arg("sigma") = 0.5f,
           "Enable Gaussian Soft NMS.  Decays overlapping box scores by "
           "exp(-iou²/sigma) instead of hard-removing them.\n"
           "sigma: decay rate (default 0.5, paper default). Smaller = "
           "stronger suppression.")
      .def("get_soft_nms", &PyModel::getSoftNms)
      .def("get_input_names", &PyModel::getInputNames)
      .def("get_output_names", &PyModel::getOutputNames)
      .def("get_last_inference_ms", &PyModel::getLastInferenceMs,
           "Time of the last inference call in milliseconds (VPSS + NPU), "
           "measured inside C++ without Python GIL overhead.")
      .def("inference_with_detections",
           &PyModel::inferenceWithDetections,
           py::arg("image"), py::arg("detections"),
           "Two-stage inference: runs the model on each face region supplied by\n"
           "'detections' (list of dicts with x1,y1,x2,y2 from a face detector).\n"
           "Returns a list of dicts in OBJECT_DETECTION_WITH_LANDMARKS format —\n"
           "one entry per detected face — so draw_detections + draw_keypoints\n"
           "both work automatically.  Landmark coordinates are remapped to\n"
           "full-frame pixel space.");

  nn.def("get_model", get_model_with_path, py::arg("model_type"),
         py::arg("model_path"), py::arg("model_config") = py::dict(),
         py::arg("device_id") = 0);
  nn.def("get_model_from_dir", get_model_with_dir, py::arg("model_type"),
         py::arg("model_dir") = "", py::arg("device_id") = 0);

  nn.def("get_model_types",
         [](const std::string& model_type_name) -> py::list {
           ModelType mt = modelTypeFromString(model_type_name);
           auto& factory = TDLModelFactory::getInstance();
           factory.loadModelConfig();
           ModelConfig cfg = factory.getModelConfig(mt);
           py::list result;
           for (const auto& t : cfg.types)
             result.append(t);
           return result;
         },
         py::arg("model_type"),
         "Return the class name list for a model type as defined in "
         "model_factory.json. Returns an empty list if no types are defined "
         "(e.g. generic YOLOV26).");

  nn.def("get_available_model_types",
         []() -> py::list {
           auto& factory = TDLModelFactory::getInstance();
           factory.loadModelConfig();
           py::list result;
           for (const auto& name : factory.getModelList())
             result.append(name);
           return result;
         },
         "Return all model type names available in model_factory.json.");

  nn.def("get_model_filename",
         [](const std::string& model_type_name) -> std::string {
           ModelType mt = modelTypeFromString(model_type_name);
           auto& factory = TDLModelFactory::getInstance();
           ModelConfig cfg = factory.getModelConfig(mt);
           auto it = cfg.custom_config_str.find("file_name");
           if (it != cfg.custom_config_str.end()) return it->second;
           return "";
         },
         py::arg("model_type"),
         "Return the base file_name for a model type as defined in model_factory.json "
         "(e.g. 'scrfd_det_face_432_768_INT8'). Build the full path by appending "
         "'_<platform>.cvimodel' and prepending the model directory.");

  // Tracker
  py::enum_<TDLObjectType>(nn, "ObjectType")
      .value("UNDEFINED", OBJECT_TYPE_UNDEFINED)
      .value("PERSON", OBJECT_TYPE_PERSON)
      .value("FACE", OBJECT_TYPE_FACE)
      .value("HAND", OBJECT_TYPE_HAND)
      .value("HEAD", OBJECT_TYPE_HEAD)
      .value("HEAD_SHOULDER", OBJECT_TYPE_HEAD_SHOULDER)
      .value("HARD_HAT", OBJECT_TYPE_HARD_HAT)
      .value("FACE_MASK", OBJECT_TYPE_FACE_MASK)
      .value("CAR", OBJECT_TYPE_CAR)
      .value("BUS", OBJECT_TYPE_BUS)
      .value("TRUCK", OBJECT_TYPE_TRUCK)
      .value("MOTORBIKE", OBJECT_TYPE_MOTORBIKE)
      .value("BICYCLE", OBJECT_TYPE_BICYCLE)
      .value("LICENSE_PLATE", OBJECT_TYPE_LICENSE_PLATE)
      .value("FIRE", OBJECT_TYPE_FIRE)
      .value("SMOKE", OBJECT_TYPE_SMOKE)
      .export_values();

  py::enum_<TrackerType>(nn, "TrackerType")
      .value("MOT_SORT", TrackerType::TDL_MOT_SORT)
      .value("SOT", TrackerType::TDL_SOT)
      .export_values();

  py::class_<TrackerConfig>(nn, "TrackerConfig")
      .def(py::init<>())
      .def_readwrite("max_unmatched_times", &TrackerConfig::max_unmatched_times_)
      .def_readwrite("track_confirmed_frames",
                     &TrackerConfig::track_confirmed_frames_)
      .def_readwrite("track_init_score_thresh",
                     &TrackerConfig::track_init_score_thresh_)
      .def_readwrite("high_score_thresh", &TrackerConfig::high_score_thresh_)
      .def_readwrite("high_score_iou_dist_thresh",
                     &TrackerConfig::high_score_iou_dist_thresh_)
      .def_readwrite("low_score_iou_dist_thresh",
                     &TrackerConfig::low_score_iou_dist_thresh_);

  py::class_<PyTracker>(nn, "Tracker")
      .def(py::init<TrackerType>(), py::arg("type") = TrackerType::TDL_MOT_SORT)
      .def("set_img_size", &PyTracker::setImgSize, py::arg("width"),
           py::arg("height"))
      .def("set_track_config", &PyTracker::setTrackConfig, py::arg("config"))
      .def("get_track_config", &PyTracker::getTrackConfig)
      .def("set_pair_config", &PyTracker::setPairConfig, py::arg("pair_map"),
           "Map of ObjectType pairs for linked tracking")
      .def("track", &PyTracker::track, py::arg("boxes"), py::arg("frame_id"),
           "Update tracker with detection boxes. Returns list of track dicts.");

  // Matcher
  py::class_<PyMatcher>(nn, "Matcher")
      .def(py::init<std::string>(), py::arg("matcher_type"),
           "Create a matcher. matcher_type: 'cosine' or 'euclidean'")
      .def("load_gallery", &PyMatcher::loadGallery, py::arg("features"),
           "Load gallery from list of numpy arrays (float32 / int8 / uint8)")
      .def("query", &PyMatcher::queryWithTopK, py::arg("features"),
           py::arg("topk") = 1,
           "Query top-k matches. Returns (indices, scores) tuple.")
      .def("update_gallery", &PyMatcher::updateGallery, py::arg("features"),
           py::arg("col"), "Update a single gallery column")
      .def("get_gallery_size", &PyMatcher::getGalleryFeatureNum)
      .def("get_feature_dim", &PyMatcher::getFeatureDim);
  /*
  py::module llm = m.def_submodule("llm", "LLM module");
  llm.def("fetch_video", &pytdl::fetch_video, py::arg("video_path"),
          py::arg("desired_fps") = 2.0, py::arg("desired_nframes") = 0,
          py::arg("max_video_sec") = 0);
  llm.def("test_fetch_video_ts", &pytdl::test_fetch_video_ts,
          py::arg("video_path"), py::arg("desired_fps") = 2.0,
          py::arg("desired_nframes") = 0, py::arg("max_video_sec") = 0);

  //   注册Qwen类
  py::class_<pytdl::PyQwen>(llm, "Qwen")
      .def(py::init<>())
      .def("model_open", &pytdl::PyQwen::modelOpen, py::arg("model_path"))
      .def("model_close", &pytdl::PyQwen::modelClose)
      .def("inference_first", &pytdl::PyQwen::inferenceFirst,
           py::arg("input_tokens"))
      .def("inference_next", &pytdl::PyQwen::inferenceNext)
      .def("inference_generate", &pytdl::PyQwen::inferenceGenerate,
           py::arg("input_tokens"), py::arg("eos_token"))
      .def("get_infer_param", &pytdl::PyQwen::getInferParam)
      .def("__enter__", [](pytdl::PyQwen& self) { return &self; })
      .def("__exit__", [](pytdl::PyQwen& self, py::object, py::object,
                          py::object) { self.modelClose(); });

  // 注册Qwen2VL类
  py::class_<pytdl::PyQwen2VL>(llm, "Qwen2VL")
      .def(py::init<>())
      .def("init", &pytdl::PyQwen2VL::init, py::arg("dev_id"),
           py::arg("model_path"))
      .def("deinit", &pytdl::PyQwen2VL::deinit)
      .def("forward_first", &pytdl::PyQwen2VL::forward_first, py::arg("tokens"),
           py::arg("position_ids"), py::arg("pixel_values"), py::arg("posids"),
           py::arg("attnmask"), py::arg("img_offset"), py::arg("pixel_num"))
      .def("forward_next", &pytdl::PyQwen2VL::forward_next)
      .def("set_generation_mode", &pytdl::PyQwen2VL::set_generation_mode,
           py::arg("mode"))
      .def("get_generation_mode", &pytdl::PyQwen2VL::get_generation_mode)
      .def_readwrite("generation_mode", &pytdl::PyQwen2VL::generation_mode)
      .def_readwrite("SEQLEN", &pytdl::PyQwen2VL::SEQLEN)
      .def_readwrite("token_length", &pytdl::PyQwen2VL::token_length)
      .def_readwrite("HIDDEN_SIZE", &pytdl::PyQwen2VL::HIDDEN_SIZE)
      .def_readwrite("NUM_LAYERS", &pytdl::PyQwen2VL::NUM_LAYERS)
      .def_readwrite("MAX_POS", &pytdl::PyQwen2VL::MAX_POS)
      .def_readwrite("MAX_PIXELS", &pytdl::PyQwen2VL::MAX_PIXELS)
      .def_readwrite("VIT_DIMS", &pytdl::PyQwen2VL::VIT_DIMS)
      .def("__enter__", [](pytdl::PyQwen2VL& self) { return &self; })
      .def("__exit__", [](pytdl::PyQwen2VL& self, py::object, py::object,
                          py::object) { self.deinit(); });
  */
  // 添加BytePairEncoder绑定
  py::module utils = m.def_submodule("utils", "Utility functions module");

  py::class_<BytePairEncoder>(utils, "BytePairEncoder")
      .def(py::init<const std::string&, const std::string&>(), "encoder_file"_a,
           "bpe_file"_a)
      .def(
          "tokenizer_bpe",
          [](BytePairEncoder& self, const std::string& text_file) {
            std::vector<std::vector<int32_t>> tokens;
            int result = self.tokenizerBPE(text_file, tokens);
            if (result != 0) {
              throw std::runtime_error("Tokenization failed");
            }
            return tokens;
          },
          "text_file"_a, "Tokenize text file and return token sequences");
}
