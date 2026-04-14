#ifndef PYTHON_MODEL_HPP_
#define PYTHON_MODEL_HPP_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "components/tracker/tracker_types.hpp"
#include "image/base_image.hpp"
#include "model/base_model.hpp"
#include "py_image.hpp"
#include "tdl_model_factory.hpp"

namespace py = pybind11;
namespace pytdl {

class PyModel {
 public:
  PyModel(std::shared_ptr<BaseModel>& model);

  py::list inference(const PyImage& image);
  py::list inference(
      const py::array_t<unsigned char, py::array::c_style>& input);
  py::list inference(const PyImage& image, const py::dict& parameters);

  virtual ~PyModel() = default;

  // Explicitly release the underlying model and its VPSS preprocessor group.
  // Call this before the script exits to avoid exhausting VPSS groups across
  // multiple script runs (the kernel does not release groups on process exit
  // unless destructors run cleanly).
  void close() { model_.reset(); }

  py::dict getPreprocessParameters();

  void setThreshold(float threshold);
  float getThreshold() const;
  void setSoftNms(bool enable, float sigma = 0.5f);
  bool getSoftNms() const;
  std::vector<std::string> getInputNames() const;
  std::vector<std::string> getOutputNames() const;

 protected:
  std::shared_ptr<BaseModel> model_;

 private:
  py::list outputParse(
      const std::vector<std::shared_ptr<ModelOutputInfo>>& out_datas);
};

class PyTracker {
 public:
  PyTracker(TrackerType type);
  void setPairConfig(
      const std::map<TDLObjectType, TDLObjectType>& object_pair_config);
  void setTrackConfig(const TrackerConfig& track_config);
  TrackerConfig getTrackConfig();
  py::list track(const py::list& boxes, uint64_t frame_id);
  void setImgSize(int width, int height);

 private:
  std::shared_ptr<Tracker> tracker_;
};

void getModelConfig(const py::dict& model_config,
                    ModelConfig& model_config_cpp);

PyModel get_model_from_dir(ModelType model_type,
                           const std::string& model_dir = "",

                           const int device_id = 0);

PyModel get_model(ModelType model_type, const std::string& model_path,
                  const py::dict& model_config = py::dict(),
                  const int device_id = 0);

}  // namespace pytdl
#endif
