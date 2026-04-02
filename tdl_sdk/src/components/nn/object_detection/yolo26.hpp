#pragma once

#include "model/base_model.hpp"

// YoloV26Detection — YOLO26 object detection for Sophgo CV181X/CV184X.
//
// YOLO26 uses a decoupled head with 4-channel box outputs (ltrb distances
// directly — no DFL / distribution focal loss), unlike YOLOv8/YOLO11 which
// use 64-channel DFL outputs.
//
// Supports two output tensor layouts:
//   Interleaved: [Box(s32), Cls(s32), Box(s16), Cls(s16), Box(s8),  Cls(s8)]
//   Grouped:     [Box(s32), Box(s16), Box(s8),  Cls(s32), Cls(s16), Cls(s8)]
// Layout is detected automatically in onModelOpened() by inspecting shapes.
//
// Supported tensor types: INT8, UINT8, FP32.

class YoloV26Detection final : public BaseModel {
 public:
  // num_cls = 0: auto-detect from model outputs.
  explicit YoloV26Detection(const int num_cls = 0);
  ~YoloV26Detection();

  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;

  virtual int32_t onModelOpened() override;
  virtual void postPreprocess(std::shared_ptr<BaseTensor> tensor,
                              int batch_idx) override;

 private:
  std::vector<int> strides_;
  std::map<int, std::string> class_out_names_;
  std::map<int, std::string> bbox_out_names_;
  std::map<int, std::string> bbox_class_out_names_;

  static constexpr int kNumBoxChannel = 4;  // ltrb, no DFL
  int num_cls_ = 0;
  float nms_threshold_ = 0.5f;

  // NHWC (sscma/YOLO26) support — same pattern as YoloV8Detection
  bool is_nhwc_input_ = false;
  int nhwc_model_h_ = 0;
  int nhwc_model_w_ = 0;
  bool nhwc_sw_norm_128_ = false;
};
