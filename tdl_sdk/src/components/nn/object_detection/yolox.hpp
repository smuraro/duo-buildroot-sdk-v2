#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloXDetection final : public BaseModel {
 public:
  YoloXDetection();
  ~YoloXDetection();

  int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;

  int32_t onModelOpened() override;
  void postPreprocess(std::shared_ptr<BaseTensor> tensor, int batch_idx) override;
  bool needsPostPreprocess() const override { return needs_packed_to_planar_; }

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int basic_pos, int grid0,
                            int grid1, std::vector<float> &decode_box);

  std::vector<int> strides;
  std::map<int, std::string> class_out_names_;
  std::map<int, std::string> object_out_names_;
  std::map<int, std::string> box_out_names_;
  int num_cls = 0;
  float nms_threshold_ = 0.5;
  bool needs_packed_to_planar_ = false;
  float model_qscale_ = 1.0f;
};
