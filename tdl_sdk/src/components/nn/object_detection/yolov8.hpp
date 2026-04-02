#pragma once
#include <bitset>

#include "model/base_model.hpp"

class YoloV8Detection final : public BaseModel {
 public:
  YoloV8Detection(const int num_cls = 0);
  YoloV8Detection(std::pair<int, int> yolov8_pair);
  ~YoloV8Detection();
  // int inference(VIDEO_FRAME_INFO_S *srcFrame, TDLObject *obj_meta)
  // override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;
  virtual void postPreprocess(std::shared_ptr<BaseTensor> tensor,
                              int batch_idx) override;

 private:
  void decodeBboxFeatureMap(int batch_idx, int stride, int anchor_idx,
                            std::vector<float> &decode_box);

  std::map<std::string, std::string> out_names_;

  // if output seperate featuremap
  std::vector<int> strides;
  std::map<int, std::string> class_out_names;
  std::map<int, std::string> bbox_out_names;
  std::map<int, std::string> bbox_class_out_names;
  int num_box_channel_ = 64;
  int num_cls_ = 0;  // would parse automatically,should not be equal with
                     // num_box_channel_
  float nms_threshold_ = 0.5;
  // NHWC (sscma/YOLO11) models: input is [N,H,W,C] and box coords are
  // normalized by model input size (not letterbox-corrected).
  bool is_nhwc_input_ = false;
  int nhwc_model_h_ = 0;
  int nhwc_model_w_ = 0;
  // True only when input dtype is INT8 (requires software -128 subtraction).
  // sscma: only subtracts 128 if (input_.type == MA_TENSOR_TYPE_S8).
  // UINT8 input means the CVI compiler folded the bias into the first layer.
  bool nhwc_sw_norm_128_ = false;
};
