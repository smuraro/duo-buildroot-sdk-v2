#include "object_detection/yolov8.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"
template <typename T>
inline void parse_cls_info(T *p_cls_ptr, int num_anchor, int num_cls,
                           int anchor_idx, int cls_offset, float qscale,
                           float *p_max_logit, int *p_max_cls) {
  int max_logit_c = -1;
  float max_logit = -1000;
  for (int c = 0; c < num_cls; c++) {
    // NCHW: data[c][h][w] = ptr[(c + cls_offset) * num_anchor + anchor_idx]
    float logit = p_cls_ptr[(c + cls_offset) * num_anchor + anchor_idx];
    if (logit > max_logit) {
      max_logit = logit;
      max_logit_c = c;
    }
  }
  *p_max_logit = max_logit * qscale;
  *p_max_cls = max_logit_c;
}

template <typename T>
inline std::vector<float> get_box_vals(T *p_box_ptr, int num_anchor,
                                       int anchor_idx, int num_box_channel,
                                       float qscale) {
  std::vector<float> box_vals;
  for (int c = 0; c < num_box_channel; c++) {
    box_vals.push_back(p_box_ptr[c * num_anchor + anchor_idx] * qscale);
  }
  return box_vals;
}

YoloV8Detection::YoloV8Detection(const int num_cls)
    : YoloV8Detection(std::make_pair(64, num_cls)) {}

YoloV8Detection::YoloV8Detection(std::pair<int, int> yolov8_pair) {
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {254.97195, 254.97195, 254.97195};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;

  num_box_channel_ = yolov8_pair.first;
  num_cls_ = yolov8_pair.second;
  if (num_box_channel_ == num_cls_) {
    LOGE("error,num_box_channel_(%d) == num_cls_(%d)", num_box_channel_,
         num_cls_);
    throw std::runtime_error("num_box_channel_ == num_cls_");
  }
}

// would parse 3 cases,1:box,cls seperate feature map,2 box+cls seperate
// featuremap,3 output decoded results
int32_t YoloV8Detection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  TensorInfo input_tensor_info = net_->getTensorInfo(input_layer);
  auto input_shape = input_tensor_info.shape;
  // Support NHWC [N,H,W,C] (sscma/YOLO11 cvimodels) and NCHW [N,C,H,W].
  // Heuristic: if the last dim is 1 or 3 it is the channel count → NHWC.
  int input_h, input_w;
  bool is_nhwc_input = (input_shape[3] == 1 || input_shape[3] == 3);
  if (is_nhwc_input) {
    input_h = input_shape[1];
    input_w = input_shape[2];
    is_nhwc_input_ = true;
    nhwc_model_h_ = input_h;
    nhwc_model_w_ = input_w;
    PreprocessParams& pp = preprocess_params_[input_layer];
    // Disable VPSS normalization — handled in postPreprocess if needed.
    for (int i = 0; i < 3; i++) {
      pp.scale[i] = 1.0f;
      pp.mean[i]  = 0.0f;
    }
    // sscma uses pure stretch (no letterbox).
    pp.keep_aspect_ratio = false;
    // Use bilinear (VPSS default is bicubic=0; bilinear=1 is close enough).
    pp.use_nearest_resize = false;
    // Only subtract 128 in postPreprocess when input dtype is INT8.
    // sscma: if (input_.type == MA_TENSOR_TYPE_S8) { subtract 128 }
    // For UINT8 inputs the CVI compiler folds the bias into the first layer.
    nhwc_sw_norm_128_ = (input_tensor_info.data_type == TDLDataType::INT8);
    LOGI("input format: NHWC [%d,%d,%d,%d] dtype=%d sw_norm_128=%d",
         input_shape[0], input_shape[1], input_shape[2], input_shape[3],
         static_cast<int>(input_tensor_info.data_type), (int)nhwc_sw_norm_128_);
  } else {
    // NCHW (standard)
    input_h = input_shape[2];
    input_w = input_shape[3];
    LOGI("input format: NCHW  [%d,%d,%d,%d]",
         input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
  }
  strides.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();

  LOGI("to parse output branch,box_channel:%d,num_cls:%d", num_box_channel_,
       num_cls_);
  if (num_cls_ == 0) {
    LOGI(
        "num_cls is 0,would take branches whose channel not equal to "
        "num_box_channel_ as cls branch");
  }
  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    int feat_h = oinfo.shape[2];
    int feat_w = oinfo.shape[3];
    int channel = oinfo.shape[1];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    if (stride_h != stride_w) {
      LOGE("stride not equal,stridew:%d,strideh:%d,featw:%d,feath:%d\n",
           stride_w, stride_h, feat_w, feat_h);
      return -1;
    }
    if (num_cls_ == 0) {
      if (channel == num_box_channel_) {
        bbox_out_names[stride_h] = output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse box branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else {
        num_cls_ = channel;
        class_out_names[stride_h] = output_layers[j];
        LOGI("parse class branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      }
    } else {
      if (channel == num_box_channel_) {
        bbox_out_names[stride_h] = output_layers[j];
        strides.push_back(stride_h);
        LOGI("parse box branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else if (channel == num_cls_) {
        class_out_names[stride_h] = output_layers[j];
        LOGI("parse class branch,name:%s,stride:%d\n", output_layers[j].c_str(),
             stride_h);
      } else if (channel == (num_box_channel_ + num_cls_)) {
        strides.push_back(stride_h);
        bbox_class_out_names[stride_h] = output_layers[j];
        LOGI("parse box+class branch,name: %s,stride:%d\n",
             output_layers[j].c_str(), stride_h);
      } else {
        LOGE("unexpected branch:%s,channel:%d\n", output_layers[j].c_str(),
             channel);
        return -1;
      }
    }
  }
  if (bbox_out_names.size() != class_out_names.size()) {
    LOGE(
        "Unsupported output type: mismatched number of box and class branches. "
        "bbox branches: %zu, class branches: %zu. "
        "Possible cause: num_classes = 64 or other unsupported configuration.",
        bbox_out_names.size(), class_out_names.size());
    return -1;
  }

  return 0;
}

YoloV8Detection::~YoloV8Detection() {}

void YoloV8Detection::postPreprocess(std::shared_ptr<BaseTensor> tensor,
                                     int batch_idx) {
  if (!is_nhwc_input_ || !nhwc_sw_norm_128_) return;
  // INT8 input: sscma subtracts 128 (pixel - 128 = int8 value).
  // uint8 arithmetic gives correct INT8 bit patterns:
  //   pixel=0   → 0   - 128 = 128 as uint8 → -128 as int8  ✓
  //   pixel=128 → 128 - 128 = 0             → 0   as int8  ✓
  //   pixel=255 → 255 - 128 = 127           → 127 as int8  ✓
  int batch_bytes = tensor->getCapacity() / tensor->getBatchSize();
  uint8_t* data = tensor->getBatchPtr<uint8_t>(batch_idx);
  for (int i = 0; i < batch_bytes; i++) {
    data[i] -= 128u;
  }
  tensor->flushCache();
}

// the bbox featuremap shape is b x 4*regmax x h   x w
void YoloV8Detection::decodeBboxFeatureMap(int batch_idx, int stride,
                                           int anchor_idx,
                                           std::vector<float> &decode_box) {
  std::string box_name;
  if (bbox_out_names.count(stride)) {
    box_name = bbox_out_names[stride];
  } else if (bbox_class_out_names.count(stride)) {
    box_name = bbox_class_out_names[stride];
  }
  TensorInfo boxinfo = net_->getTensorInfo(box_name);
  std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

  int num_channel = boxinfo.shape[1];
  int num_anchor = boxinfo.shape[2] * boxinfo.shape[3];
  int box_val_num = 4;
  int reg_max = 16;
  if (num_box_channel_ != box_val_num * reg_max) {
    LOGE("box channel size not ok,got:%d\n", num_channel);
  }

  int32_t feat_w = boxinfo.shape[3];

  int anchor_y = anchor_idx / feat_w;
  int anchor_x = anchor_idx % feat_w;

  float grid_y = anchor_y + 0.5;
  float grid_x = anchor_x + 0.5;

  std::vector<float> grid_logits;  // 4x16
  float qscale = boxinfo.qscale;
  if (boxinfo.data_type == TDLDataType::INT8) {
    int8_t *p_box_int8 = box_tensor->getBatchPtr<int8_t>(batch_idx);
    grid_logits = get_box_vals(p_box_int8, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    grid_logits = get_box_vals(p_box_uint8, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    grid_logits = get_box_vals(p_box_float, num_anchor, anchor_idx,
                               num_box_channel_, qscale);
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }

  // compute softmax and accumulate val per 16
  std::vector<float> box_vals;
  for (int i = 0; i < box_val_num; i++) {
    float sum_softmax = 0;
    float sum_val = 0;
    for (int j = 0; j < reg_max; j++) {
      float expv = exp(grid_logits[i * reg_max + j]);
      sum_softmax += expv;
      sum_val += expv * j;
    }
    sum_softmax = sum_val / sum_softmax;
    box_vals.push_back(sum_softmax);
  }

  std::vector<float> box = {
      (grid_x - box_vals[0]) * stride, (grid_y - box_vals[1]) * stride,
      (grid_x + box_vals[2]) * stride, (grid_y + box_vals[3]) * stride};
  decode_box = box;
}

int32_t YoloV8Detection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  // Same NHWC/NCHW detection as onModelOpened()
  uint32_t input_width, input_height;
  if (input_tensor.shape[3] == 1 || input_tensor.shape[3] == 3) {
    input_height = input_tensor.shape[1];
    input_width  = input_tensor.shape[2];
  } else {
    input_height = input_tensor.shape[2];
    input_width  = input_tensor.shape[3];
  }
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  float inverse_th = std::log(model_threshold_ / (1 - model_threshold_));
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d,model "
      "threshold:%f,inverse th:%f",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_,
      inverse_th);

  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];

      // ── cls tensor (fetched once per stride) ────────────────────────────
      std::string cls_name;
      int cls_offset = 0;
      if (class_out_names.count(stride)) {
        cls_name = class_out_names[stride];
      } else if (bbox_class_out_names.count(stride)) {
        cls_name = bbox_class_out_names[stride];
        cls_offset = num_box_channel_;
      }
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);
      int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
      int num_cls    = num_cls_;
      int num_anchor = classinfo.shape[2] * classinfo.shape[3];
      float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;

      // ── box tensor (fetched once per stride) ────────────────────────────
      std::string box_name = bbox_out_names.count(stride)
                                 ? bbox_out_names.at(stride)
                                 : bbox_class_out_names.at(stride);
      TensorInfo boxinfo = net_->getTensorInfo(box_name);
      std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);
      int box_feat_w    = boxinfo.shape[3];
      int box_num_anchor = boxinfo.shape[2] * boxinfo.shape[3];
      float box_qscale  = boxinfo.qscale;

      // For sscma/NHWC models: use BOX qscale for cls scores
      if (is_nhwc_input_ && box_qscale > 0) cls_qscale = box_qscale;

      LOGI("stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d,qscale:%f\n",
           stride, classinfo.shape[3], classinfo.shape[2],
           num_per_pixel, num_cls, cls_qscale);

      // ── batch pointers (fetched once per stride, not per anchor) ────────
      void *cls_ptr = nullptr;
      if (classinfo.data_type == TDLDataType::INT8)
        cls_ptr = cls_tensor->getBatchPtr<int8_t>(b);
      else if (classinfo.data_type == TDLDataType::UINT8)
        cls_ptr = cls_tensor->getBatchPtr<uint8_t>(b);
      else if (classinfo.data_type == TDLDataType::FP32)
        cls_ptr = cls_tensor->getBatchPtr<float>(b);
      else {
        LOGE("unsupported cls data type:%d\n",
             static_cast<int>(classinfo.data_type));
        continue;
      }

      void *box_ptr = nullptr;
      if (boxinfo.data_type == TDLDataType::INT8)
        box_ptr = box_tensor->getBatchPtr<int8_t>(b);
      else if (boxinfo.data_type == TDLDataType::UINT8)
        box_ptr = box_tensor->getBatchPtr<uint8_t>(b);
      else if (boxinfo.data_type == TDLDataType::FP32)
        box_ptr = box_tensor->getBatchPtr<float>(b);
      else {
        LOGE("unsupported box data type:%d\n",
             static_cast<int>(boxinfo.data_type));
        continue;
      }

      for (int j = 0; j < num_anchor; j++) {
        // ── class score ───────────────────────────────────────────────────
        int   max_logit_c = -1;
        float max_logit   = -1000;
        if (classinfo.data_type == TDLDataType::INT8)
          parse_cls_info(static_cast<int8_t *>(cls_ptr), num_anchor, num_cls,
                         j, cls_offset, cls_qscale, &max_logit, &max_logit_c);
        else if (classinfo.data_type == TDLDataType::UINT8)
          parse_cls_info(static_cast<uint8_t *>(cls_ptr), num_anchor, num_cls,
                         j, cls_offset, cls_qscale, &max_logit, &max_logit_c);
        else
          parse_cls_info(static_cast<float *>(cls_ptr), num_anchor, num_cls,
                         j, cls_offset, cls_qscale, &max_logit, &max_logit_c);

        if (max_logit < inverse_th) continue;

        // ── DFL box decode (inline, no extra lookup) ──────────────────────
        float grid_x = static_cast<float>(j % box_feat_w) + 0.5f;
        float grid_y = static_cast<float>(j / box_feat_w) + 0.5f;

        std::vector<float> grid_logits;
        if (boxinfo.data_type == TDLDataType::INT8)
          grid_logits = get_box_vals(static_cast<int8_t *>(box_ptr),
                                     box_num_anchor, j, num_box_channel_,
                                     box_qscale);
        else if (boxinfo.data_type == TDLDataType::UINT8)
          grid_logits = get_box_vals(static_cast<uint8_t *>(box_ptr),
                                     box_num_anchor, j, num_box_channel_,
                                     box_qscale);
        else
          grid_logits = get_box_vals(static_cast<float *>(box_ptr),
                                     box_num_anchor, j, num_box_channel_,
                                     1.0f);

        std::vector<float> box_vals;
        for (int k = 0; k < 4; k++) {
          float sum_s = 0, sum_v = 0;
          for (int r = 0; r < 16; r++) {
            float e = std::exp(grid_logits[k * 16 + r]);
            sum_s += e;
            sum_v += e * r;
          }
          box_vals.push_back(sum_v / sum_s);
        }

        float score = 1.0f / (1.0f + std::exp(-max_logit));
        ObjectBoxInfo bbox;
        bbox.score    = score;
        bbox.x1 = std::max(0.0f, std::min((grid_x - box_vals[0]) * stride, input_width_f));
        bbox.y1 = std::max(0.0f, std::min((grid_y - box_vals[1]) * stride, input_height_f));
        bbox.x2 = std::max(0.0f, std::min((grid_x + box_vals[2]) * stride, input_width_f));
        bbox.y2 = std::max(0.0f, std::min((grid_y + box_vals[3]) * stride, input_height_f));
        bbox.class_id = max_logit_c;
        LOGI("bbox:[%f,%f,%f,%f],score:%f,label:%d,logit:%f\n", bbox.x1,
             bbox.y1, bbox.x2, bbox.y2, bbox.score, max_logit_c, max_logit);

        lb_boxes[max_logit_c].push_back(bbox);
      }
    }
    if (use_soft_nms_)
      DetectionHelper::softNmsObjects(lb_boxes, model_threshold_, soft_nms_sigma_);
    else
      DetectionHelper::nmsObjects(lb_boxes, nms_threshold_);
    std::vector<float> scale_params =
        batch_rescale_params_[input_tensor_name][b];

    std::shared_ptr<ModelBoxInfo> obj = std::make_shared<ModelBoxInfo>();
    obj->image_width = image_width;
    obj->image_height = image_height;
    for (auto &bbox : lb_boxes) {
      for (auto &b : bbox.second) {
        DetectionHelper::rescaleBbox(b, scale_params);
        if (type_mapping_.count(b.class_id)) {
          b.object_type = type_mapping_[b.class_id];
        }
        obj->bboxes.push_back(b);
      }
    }
    out_datas.push_back(obj);
  }
  return 0;
}
