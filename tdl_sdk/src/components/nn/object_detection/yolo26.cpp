#include "object_detection/yolo26.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

// ─── helpers ─────────────────────────────────────────────────────────────────

template <typename T>
static inline void parse_cls_info26(T *ptr, int num_anchor, int num_cls,
                                    int anchor_idx, float qscale,
                                    float *p_max_logit, int *p_max_cls) {
  int best_c = -1;
  float best = -1000.0f;
  for (int c = 0; c < num_cls; c++) {
    float v = static_cast<float>(ptr[c * num_anchor + anchor_idx]) * qscale;
    if (v > best) { best = v; best_c = c; }
  }
  *p_max_logit = best;
  *p_max_cls   = best_c;
}

template <typename T>
static inline std::vector<float> read_box_vals26(T *ptr, int num_anchor,
                                                  int anchor_idx,
                                                  float qscale) {
  std::vector<float> v(4);
  for (int c = 0; c < 4; c++)
    v[c] = static_cast<float>(ptr[c * num_anchor + anchor_idx]) * qscale;
  return v;
}

// ─── constructor / destructor ────────────────────────────────────────────────

YoloV26Detection::YoloV26Detection(const int num_cls) {
  net_param_.model_config.mean     = {0.0f, 0.0f, 0.0f};
  net_param_.model_config.std      = {254.97195f, 254.97195f, 254.97195f};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;
  num_cls_ = num_cls;
}

YoloV26Detection::~YoloV26Detection() {}

// ─── onModelOpened ───────────────────────────────────────────────────────────
//
// Classifies each output tensor as box (channel==4) or cls (channel!=4).
// Works for both interleaved and grouped layouts because we key by stride.

int32_t YoloV26Detection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  TensorInfo input_tensor_info = net_->getTensorInfo(input_layer);
  auto input_shape = input_tensor_info.shape;

  // Detect NHWC input: shape[3] == 3 or 1 means [N,H,W,C]
  bool is_nhwc = (input_shape.size() == 4 &&
                  (input_shape[3] == 3 || input_shape[3] == 1));
  int input_h, input_w;
  if (is_nhwc) {
    input_h = input_shape[1];
    input_w = input_shape[2];
    is_nhwc_input_ = true;
    nhwc_model_h_  = input_h;
    nhwc_model_w_  = input_w;
    PreprocessParams &pp = preprocess_params_[input_layer];
    for (int i = 0; i < 3; i++) {
      pp.scale[i] = 1.0f;
      pp.mean[i]  = 0.0f;
    }
    pp.keep_aspect_ratio  = false;
    pp.use_nearest_resize = false;
    nhwc_sw_norm_128_ = (input_tensor_info.data_type == TDLDataType::INT8);
    LOGI("YoloV26 NHWC input [%d,%d,%d,%d] dtype=%d sw_norm_128=%d",
         input_shape[0], input_shape[1], input_shape[2], input_shape[3],
         static_cast<int>(input_tensor_info.data_type), nhwc_sw_norm_128_);
  } else {
    input_h = input_shape[2];
    input_w = input_shape[3];
  }

  strides_.clear();
  bbox_out_names_.clear();
  class_out_names_.clear();
  bbox_class_out_names_.clear();

  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  LOGI("YoloV26 onModelOpened: %zu outputs, num_cls=%d", num_output, num_cls_);

  for (size_t j = 0; j < num_output; j++) {
    auto oinfo   = net_->getTensorInfo(output_layers[j]);
    int feat_h   = oinfo.shape[2];
    int feat_w   = oinfo.shape[3];
    int channel  = oinfo.shape[1];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    LOGI("  output %s: shape=[%s]  feat=%dx%d  stride=%dx%d",
         output_layers[j].c_str(),
         [&]() {
           std::string s;
           for (int d : oinfo.shape)
             s += std::to_string(d) + ",";
           return s;
         }().c_str(),
         feat_h, feat_w, stride_h, stride_w);

    if (stride_h != stride_w) {
      LOGE("YoloV26: non-square stride (%d vs %d) for output %s",
           stride_h, stride_w, output_layers[j].c_str());
      return -1;
    }

    if (channel == kNumBoxChannel) {
      // Box tensor: 4 channels = ltrb distances
      bbox_out_names_[stride_h] = output_layers[j];
      strides_.push_back(stride_h);
      LOGI("  box  branch: %s  stride=%d  channel=%d",
           output_layers[j].c_str(), stride_h, channel);
    } else {
      // Class tensor
      if (num_cls_ == 0) num_cls_ = channel;
      class_out_names_[stride_h] = output_layers[j];
      LOGI("  cls  branch: %s  stride=%d  channel=%d",
           output_layers[j].c_str(), stride_h, channel);
    }
  }

  if (bbox_out_names_.size() != class_out_names_.size()) {
    LOGE("YoloV26: box/cls branch count mismatch (%zu vs %zu)",
         bbox_out_names_.size(), class_out_names_.size());
    return -1;
  }

  if (strides_.empty()) {
    LOGE("YoloV26: no box branches found");
    return -1;
  }

  LOGI("YoloV26 onModelOpened done: %zu scales, num_cls=%d",
       strides_.size(), num_cls_);
  return 0;
}

// ─── postPreprocess ──────────────────────────────────────────────────────────
//
// For INT8 NHWC models only: subtract 128 so uint8 pixel values become int8.
// UINT8 models have this bias folded into the first layer by the CVI compiler.

void YoloV26Detection::postPreprocess(std::shared_ptr<BaseTensor> tensor,
                                      int batch_idx) {
  if (!is_nhwc_input_ || !nhwc_sw_norm_128_) return;
  int batch_bytes = tensor->getCapacity() / tensor->getBatchSize();
  uint8_t *data   = tensor->getBatchPtr<uint8_t>(batch_idx);
  for (int i = 0; i < batch_bytes; i++) data[i] -= 128u;
  tensor->flushCache();
}

// decodeBboxFeatureMap removed — box decoding inlined in outputParse per stride
// to avoid getTensorInfo / getOutputTensor calls inside the per-anchor loop.

// ─── outputParse ─────────────────────────────────────────────────────────────

int32_t YoloV26Detection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {

  const std::string &input_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_name);
  float input_w_f, input_h_f;
  if (is_nhwc_input_) {
    input_h_f = static_cast<float>(nhwc_model_h_);
    input_w_f = static_cast<float>(nhwc_model_w_);
  } else {
    input_h_f = static_cast<float>(input_tensor.shape[2]);
    input_w_f = static_cast<float>(input_tensor.shape[3]);
  }
  float inverse_th = std::log(model_threshold_ / (1.0f - model_threshold_));

  LOGI("YoloV26 outputParse: batch=%d  input=%dx%d  threshold=%.3f",
       static_cast<int>(images.size()),
       input_tensor.shape[3], input_tensor.shape[2], model_threshold_);

  for (int b = 0; b < static_cast<int>(input_tensor.shape[0]); b++) {
    uint32_t image_w = images[b]->getWidth();
    uint32_t image_h = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;

    for (int stride : strides_) {
      // ── cls tensor (fetched once per stride) ──────────────────────────────
      const std::string &cls_name = class_out_names_.count(stride)
                                        ? class_out_names_.at(stride)
                                        : bbox_class_out_names_.at(stride);
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);

      int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
      int num_cls       = num_cls_;
      int num_anchor    = classinfo.shape[2] * classinfo.shape[3];
      int feat_w_cls    = classinfo.shape[3];
      float cls_qscale  = (num_per_pixel == 1) ? classinfo.qscale : 1.0f;

      // ── box tensor (fetched once per stride) ──────────────────────────────
      const std::string &box_name = bbox_out_names_.count(stride)
                                        ? bbox_out_names_.at(stride)
                                        : bbox_class_out_names_.at(stride);
      TensorInfo boxinfo = net_->getTensorInfo(box_name);
      std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

      int box_feat_w = boxinfo.shape[3];
      float box_qscale = boxinfo.qscale;

      LOGI("  stride=%d  feat=%dx%d  num_cls=%d  cls_qscale=%.6f  box_qscale=%.6f",
           stride, classinfo.shape[3], classinfo.shape[2],
           num_cls, cls_qscale, box_qscale);

      for (int j = 0; j < num_anchor; j++) {
        // ── class score ─────────────────────────────────────────────────────
        int   max_cls   = -1;
        float max_logit = -1000.0f;

        if (classinfo.data_type == TDLDataType::INT8) {
          parse_cls_info26(cls_tensor->getBatchPtr<int8_t>(b),
                           num_anchor, num_cls, j, cls_qscale,
                           &max_logit, &max_cls);
        } else if (classinfo.data_type == TDLDataType::UINT8) {
          parse_cls_info26(cls_tensor->getBatchPtr<uint8_t>(b),
                           num_anchor, num_cls, j, cls_qscale,
                           &max_logit, &max_cls);
        } else if (classinfo.data_type == TDLDataType::FP32) {
          parse_cls_info26(cls_tensor->getBatchPtr<float>(b),
                           num_anchor, num_cls, j, 1.0f,
                           &max_logit, &max_cls);
        } else {
          LOGE("YoloV26: unsupported cls data type %d",
               static_cast<int>(classinfo.data_type));
          continue;
        }

        if (max_logit < inverse_th) continue;

        // ── box decode (inline, no extra lookup) ────────────────────────────
        float grid_y = static_cast<float>(j / box_feat_w) + 0.5f;
        float grid_x = static_cast<float>(j % box_feat_w) + 0.5f;

        std::vector<float> ltrb;
        if (boxinfo.data_type == TDLDataType::INT8) {
          ltrb = read_box_vals26(box_tensor->getBatchPtr<int8_t>(b),
                                 num_anchor, j, box_qscale);
        } else if (boxinfo.data_type == TDLDataType::UINT8) {
          ltrb = read_box_vals26(box_tensor->getBatchPtr<uint8_t>(b),
                                 num_anchor, j, box_qscale);
        } else if (boxinfo.data_type == TDLDataType::FP32) {
          ltrb = read_box_vals26(box_tensor->getBatchPtr<float>(b),
                                 num_anchor, j, 1.0f);
        } else {
          LOGE("YoloV26: unsupported box data type %d",
               static_cast<int>(boxinfo.data_type));
          continue;
        }

        float score = 1.0f / (1.0f + std::exp(-max_logit));

        ObjectBoxInfo bbox;
        bbox.score    = score;
        bbox.x1 = std::max(0.0f, std::min((grid_x - ltrb[0]) * stride, input_w_f));
        bbox.y1 = std::max(0.0f, std::min((grid_y - ltrb[1]) * stride, input_h_f));
        bbox.x2 = std::max(0.0f, std::min((grid_x + ltrb[2]) * stride, input_w_f));
        bbox.y2 = std::max(0.0f, std::min((grid_y + ltrb[3]) * stride, input_h_f));
        bbox.class_id = max_cls;

        lb_boxes[max_cls].push_back(bbox);
      }
    }

    if (use_soft_nms_)
      DetectionHelper::softNmsObjects(lb_boxes, model_threshold_, soft_nms_sigma_);
    else
      DetectionHelper::nmsObjects(lb_boxes, nms_threshold_);

    const auto &scale_params = batch_rescale_params_[input_name][b];
    auto obj = std::make_shared<ModelBoxInfo>();
    obj->image_width  = image_w;
    obj->image_height = image_h;

    for (auto &cls_entry : lb_boxes) {
      for (auto &det : cls_entry.second) {
        DetectionHelper::rescaleBbox(det, scale_params);
        if (type_mapping_.count(det.class_id))
          det.object_type = type_mapping_.at(det.class_id);
        obj->bboxes.push_back(det);
      }
    }
    out_datas.push_back(obj);
  }

  return 0;
}
