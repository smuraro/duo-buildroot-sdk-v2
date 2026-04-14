#ifndef PYTHON_CAMERA_HPP_
#define PYTHON_CAMERA_HPP_
#include <pybind11/pybind11.h>
#include <memory>
#include "components/video_decoder/video_decoder_type.hpp"
#include "image/base_image.hpp"
#include "py_image.hpp"

namespace py = pybind11;
namespace pytdl {

class PyCamera {
 public:
  PyCamera(int32_t width, int32_t height,
           ImageFormat format = ImageFormat::YUV420SP_VU,
           int32_t vb_buffer_num = 3,
           bool mirror = false, bool flip = false);
  ~PyCamera();

  PyImage read(int32_t channel = 0);
  int32_t release(int32_t channel = 0);
  void close();

  PyCamera* enter() { return this; }
  void exit(py::object, py::object, py::object) { close(); }

 private:
  std::shared_ptr<VideoDecoder> decoder_;
  bool closed_ = false;
};

}  // namespace pytdl
#endif
