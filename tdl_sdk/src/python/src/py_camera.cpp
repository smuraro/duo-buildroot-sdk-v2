#include "py_camera.hpp"
#include "components/video_decoder/video_decoder_type.hpp"
#include "utils/tdl_log.hpp"
#include <unistd.h>

namespace pytdl {

PyCamera::PyCamera(int32_t width, int32_t height, ImageFormat format,
                   int32_t vb_buffer_num, bool mirror, bool flip) {
  decoder_ = VideoDecoderFactory::createVideoDecoder(VideoDecoderType::VI);
  if (!decoder_) {
    throw std::runtime_error("Failed to create VI video decoder");
  }

  int32_t ret = decoder_->init("", {});
  if (ret != 0) {
    throw std::runtime_error("VideoDecoder init failed, ret: " +
                             std::to_string(ret));
  }

  ret = decoder_->initialize(width, height, format, vb_buffer_num, mirror, flip);
  if (ret != 0) {
    throw std::runtime_error("VideoDecoder initialize failed, ret: " +
                             std::to_string(ret));
  }

  // Discard the first few frames so the ISP/sensor has time to stabilize
  // (exposure, white balance and gain converge over the initial frames).
  static constexpr int kWarmupFrames = 5;
  for (int i = 0; i < kWarmupFrames; i++) {
    std::shared_ptr<BaseImage> dummy;
    if (decoder_->read(dummy, 0) == 0) {
      decoder_->release(0);
    }
  }
  LOGI("Camera warm-up done (%d frames discarded)\n", kWarmupFrames);
}

PyCamera::~PyCamera() { close(); }

PyImage PyCamera::read(int32_t channel) {
  if (closed_) {
    throw std::runtime_error("Camera is closed");
  }
  std::shared_ptr<BaseImage> image;
  int32_t ret = decoder_->read(image, channel);
  if (ret != 0 || !image) {
    throw std::runtime_error("Failed to read frame from camera, ret: " +
                             std::to_string(ret));
  }
  return PyImage(image);
}

int32_t PyCamera::release(int32_t channel) {
  if (closed_) {
    throw std::runtime_error("Camera is closed");
  }
  return decoder_->release(channel);
}

void PyCamera::close() {
  if (!closed_ && decoder_) {
    // VideoDecoder destructor (VPSS/VI Destroy*) can block — release the
    // GIL so the Python main thread / watchdog can run during this time.
    closed_ = true;
    py::gil_scoped_release nogil;
    decoder_.reset();
  }
}

}  // namespace pytdl
