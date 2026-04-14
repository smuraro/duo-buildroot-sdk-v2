#include "object_detection/yolox.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "utils/detection_helper.hpp"
#include "utils/tdl_log.hpp"

float yolox_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

template <typename T>
void get_box_vals(T *ptr, float qscale, int basic_pos, int grid0, int grid1,
                  int stride, std::vector<float> &decode_box) {
  // 计算中心点坐标和宽高
  float x_center = (ptr[basic_pos + 0] * qscale + grid0) * stride;
  float y_center = (ptr[basic_pos + 1] * qscale + grid1) * stride;
  float w = std::exp(ptr[basic_pos + 2] * qscale) * stride;
  float h = std::exp(ptr[basic_pos + 3] * qscale) * stride;

  float x0 = x_center - w * 0.5f;
  float y0 = y_center - h * 0.5f;
  float x1 = x0 + w;
  float y1 = y0 + h;

  // 清空并写入 decode_box
  decode_box.clear();
  decode_box.push_back(x0);
  decode_box.push_back(y0);
  decode_box.push_back(x1);
  decode_box.push_back(y1);
}

void YoloXDetection::decodeBboxFeatureMap(int batch_idx, int stride,
                                          int basic_pos, int grid0, int grid1,
                                          std::vector<float> &decode_box) {
  std::string box_name;
  if (box_out_names_.count(stride)) {
    box_name = box_out_names_[stride];
  } else {
    LOGE("No box name found for stride %d\n", stride);
    return;
  }

  TensorInfo boxinfo = net_->getTensorInfo(box_name);
  std::shared_ptr<BaseTensor> box_tensor = net_->getOutputTensor(box_name);

  float qscale = boxinfo.qscale;

  if (boxinfo.data_type == TDLDataType::INT8) {
    int8_t *p_box_int8 = box_tensor->getBatchPtr<int8_t>(batch_idx);
    get_box_vals(p_box_int8, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else if (boxinfo.data_type == TDLDataType::UINT8) {
    uint8_t *p_box_uint8 = box_tensor->getBatchPtr<uint8_t>(batch_idx);
    get_box_vals(p_box_uint8, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else if (boxinfo.data_type == TDLDataType::FP32) {
    float *p_box_float = box_tensor->getBatchPtr<float>(batch_idx);
    get_box_vals(p_box_float, qscale, basic_pos, grid0, grid1, stride,
                 decode_box);
  } else {
    LOGE("unsupported data type:%d\n", static_cast<int>(boxinfo.data_type));
    return;
  }
}

template <typename T>
int yolox_argmax(T *ptr, int basic_pos, int cls_len) {
  int max_idx = 0;
  for (int i = 0; i < cls_len; i++) {
    if (ptr[i + basic_pos] > ptr[max_idx + basic_pos]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int32_t YoloXDetection::outputParse(
    const std::vector<std::shared_ptr<BaseImage>> &images,
    std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) {
  std::string input_tensor_name = net_->getInputNames()[0];
  TensorInfo input_tensor = net_->getTensorInfo(input_tensor_name);
  uint32_t input_width  = input_tensor.shape[3];
  uint32_t input_height = input_tensor.shape[2];
  float input_width_f = float(input_width);
  float input_height_f = float(input_height);
  LOGI(
      "outputParse,batch size:%d,input shape:%d,%d,%d,%d,model "
      "threshold:%f",
      images.size(), input_tensor.shape[0], input_tensor.shape[1],
      input_tensor.shape[2], input_tensor.shape[3], model_threshold_);

  std::stringstream ss;
  for (uint32_t b = 0; b < (uint32_t)input_tensor.shape[0]; b++) {
    uint32_t image_width = images[b]->getWidth();
    uint32_t image_height = images[b]->getHeight();

    std::map<int, std::vector<ObjectBoxInfo>> lb_boxes;
    for (size_t i = 0; i < strides.size(); i++) {
      int stride = strides[i];
      std::string cls_name = class_out_names_[stride];
      TensorInfo classinfo = net_->getTensorInfo(cls_name);
      std::shared_ptr<BaseTensor> cls_tensor = net_->getOutputTensor(cls_name);
      int num_cls = classinfo.shape[3];

      std::string obj_name = object_out_names_[stride];
      TensorInfo objectinfo = net_->getTensorInfo(obj_name);
      std::shared_ptr<BaseTensor> obj_tensor = net_->getOutputTensor(obj_name);

      int num_grid_w = input_width / stride;
      int num_grid_h = input_height / stride;

      int basic_pos_class = 0;
      int basic_pos_object = 0;
      int basic_pos_box = 0;

      for (int g1 = 0; g1 < num_grid_h; g1++) {
        for (int g0 = 0; g0 < num_grid_w; g0++) {
          float class_score = 0.0f;
          float box_objectness = 0.0f;
          int label = 0;

          if (objectinfo.data_type == TDLDataType::INT8) {
            box_objectness =
                obj_tensor->getBatchPtr<int8_t>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else if (objectinfo.data_type == TDLDataType::UINT8) {
            box_objectness =
                obj_tensor->getBatchPtr<uint8_t>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else if (objectinfo.data_type == TDLDataType::FP32) {
            box_objectness =
                obj_tensor->getBatchPtr<float>(b)[basic_pos_object] *
                objectinfo.qscale;
          } else {
            LOGE("unsupported data type:%d\n",
                 static_cast<int>(objectinfo.data_type));
            assert(0);
          }

          if (classinfo.data_type == TDLDataType::INT8) {
            label = yolox_argmax<int8_t>(cls_tensor->getBatchPtr<int8_t>(b),
                                         basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<int8_t>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else if (classinfo.data_type == TDLDataType::UINT8) {
            label = yolox_argmax<uint8_t>(cls_tensor->getBatchPtr<uint8_t>(b),
                                          basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<uint8_t>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else if (classinfo.data_type == TDLDataType::FP32) {
            label = yolox_argmax<float>(cls_tensor->getBatchPtr<float>(b),
                                        basic_pos_class, num_cls);
            class_score =
                cls_tensor->getBatchPtr<float>(b)[basic_pos_class + label] *
                classinfo.qscale;
          } else {
            LOGE("unsupported data type:%d\n",
                 static_cast<int>(classinfo.data_type));
            assert(0);
          }

          box_objectness = yolox_sigmoid(box_objectness);
          class_score = yolox_sigmoid(class_score);
          float box_prob = box_objectness * class_score;
          if (box_prob < model_threshold_) {
            basic_pos_class += num_cls;
            basic_pos_box += 4;
            basic_pos_object += 1;
            continue;
          }
          std::vector<float> box;
          decodeBboxFeatureMap(b, stride, basic_pos_box, g0, g1, box);
          ObjectBoxInfo bbox;
          bbox.score = class_score;
          bbox.x1 = std::max(0.0f, std::min(box[0], input_width_f));
          bbox.y1 = std::max(0.0f, std::min(box[1], input_height_f));
          bbox.x2 = std::max(0.0f, std::min(box[2], input_width_f));
          bbox.y2 = std::max(0.0f, std::min(box[3], input_height_f));
          bbox.class_id = label;
          LOGI("bbox:[%f,%f,%f,%f],score:%f,label:%d:%f\n", bbox.x1, bbox.y1,
               bbox.x2, bbox.y2, bbox.score, label);

          lb_boxes[label].push_back(bbox);

          basic_pos_class += num_cls;
          basic_pos_box += 4;
          basic_pos_object += 1;
        }
      }
    }
    if (use_soft_nms_)
      DetectionHelper::softNmsObjects(lb_boxes, model_threshold_, soft_nms_sigma_);
    else
      DetectionHelper::nmsObjects(lb_boxes, nms_threshold_);
    std::vector<float> scale_params =
        batch_rescale_params_[input_tensor_name][b];
    LOGI("scale_params:%f,%f,%f,%f", scale_params[0], scale_params[1],
         scale_params[2], scale_params[3]);
    ss << "batch:" << b << "\n";

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
        ss << "bbox:[" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2
           << "],score:" << b.score << ",label:" << bbox.first << "\n";
      }
    }
    out_datas.push_back(obj);
  }
  LOGI("outputParse done,ss:%s", ss.str().c_str());
  return 0;
}

YoloXDetection::YoloXDetection() {
  // VPSS formula: INT8 = round(pixel * (1/std) * qscale)
  // input qscale=127, pixels [0,255], INT8 range [-128,127].
  // To map pixel 255 → INT8 127: (1/std)*127 = 127/255 → std = 255.0
  // This gives: pixel 128 → INT8 64, pixel 255 → INT8 127.
  net_param_.model_config.mean = {0.0, 0.0, 0.0};
  net_param_.model_config.std = {255.0, 255.0, 255.0};
  net_param_.model_config.rgb_order = "rgb";
  keep_aspect_ratio_ = true;
}

int YoloXDetection::onModelOpened() {
  const auto &input_layer = net_->getInputNames()[0];
  TensorInfo input_info = net_->getTensorInfo(input_layer);
  auto input_shape = input_info.shape;
  int input_h = input_shape[2];
  int input_w = input_shape[3];

  // CV181X VPSS does not support PIXEL_FORMAT_RGB_888_PLANAR with INT8 output.
  // Override preprocess_params_ to use RGB_PACKED UINT8 (which VPSS supports),
  // then postPreprocess deinterleaves packed→planar and quantizes to INT8.
  // YOLOX was compiled with --quant_input and scale=1/255, mean=0:
  //   int8 = round(pixel * (1/255) / qscale) where qscale≈1/127
  //   → int8 = round(pixel * 127 / 255) = round(pixel * 0.498)
  //   pixel 0→0, pixel 128→63, pixel 255→127
  if (input_info.data_type == TDLDataType::INT8) {
    PreprocessParams& pp = preprocess_params_[input_layer];
    pp.dst_image_format = ImageFormat::RGB_PACKED;
    pp.dst_pixdata_type = TDLDataType::UINT8;  // VPSS writes UINT8 packed
    pp.dst_width  = input_w;
    pp.dst_height = input_h;
    pp.keep_aspect_ratio = keep_aspect_ratio_;
    pp.scale[0] = pp.scale[1] = pp.scale[2] = 1.0f;
    pp.mean[0]  = pp.mean[1]  = pp.mean[2]  = 0.0f;
    needs_packed_to_planar_ = true;
    model_qscale_ = input_info.qscale;  // 127
    LOGI("YoloX: RGB_PACKED UINT8 override, qscale=%.4f", input_info.qscale);
  }

  strides.clear();
  const auto &output_layers = net_->getOutputNames();
  size_t num_output = output_layers.size();
  LOGI("onModelOpened: input=%dx%d, num_outputs=%zu\n", input_w, input_h, num_output);

  for (size_t j = 0; j < num_output; j++) {
    auto oinfo = net_->getTensorInfo(output_layers[j]);
    LOGI("  output[%zu] %s shape=[%d,%d,%d,%d]\n", j, output_layers[j].c_str(),
         oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
    // Model outputs use NHWC layout: [batch, H, W, C]
    int feat_h  = oinfo.shape[1];
    int feat_w  = oinfo.shape[2];
    int channel = oinfo.shape[3];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    // Identify tensor type by channel count instead of positional index:
    //   4 channels  → box regression output
    //   1 channel   → objectness output
    //   other       → class score output
    if (channel == 4) {
      box_out_names_[stride_h] = output_layers[j];
      LOGI("box feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
    } else if (channel == 1) {
      object_out_names_[stride_h] = output_layers[j];
      LOGI("object feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
    } else {
      class_out_names_[stride_h] = output_layers[j];
      LOGI("class feature %s: (%d %d %d %d)\n", output_layers[j].c_str(),
           oinfo.shape[0], oinfo.shape[1], oinfo.shape[2], oinfo.shape[3]);
      strides.push_back(stride_h);
    }
  }
  for (size_t i = 0; i < strides.size(); i++) {
    if (!class_out_names_.count(strides[i]) ||
        !box_out_names_.count(strides[i]) ||
        !object_out_names_.count(strides[i])) {
      return -1;
    }
  }

  return 0;
}

void YoloXDetection::postPreprocess(std::shared_ptr<BaseTensor> tensor,
                                    int batch_idx) {
  if (!needs_packed_to_planar_) return;

  int H = tensor->getShape()[2];
  int W = tensor->getShape()[3];
  int plane = H * W;

  uint8_t* buf = tensor->getBatchPtr<uint8_t>(batch_idx);

  // Log first 16 bytes to see what copyFromImage actually wrote

  std::vector<uint8_t> tmp(plane * 3);
  std::memcpy(tmp.data(), buf, plane * 3);

  // TPU-MLIR VPSS formula: int8 = round(pixel * (1/std) * qscale)
  // YoloV8 uses std=254.97, qscale≈127 → scale = 127/255 ≈ 0.498 (works)
  // YOLOX was compiled the same way → same scale applies.
  // pixel=255 → int8=127, pixel=128 → int8=63, pixel=0 → int8=0
  const float scale = model_qscale_ / 255.0f;
  int8_t* dst = reinterpret_cast<int8_t*>(buf);
  for (int c = 0; c < 3; c++) {
    for (int i = 0; i < plane; i++) {
      float v = tmp[i * 3 + c] * scale;
      int iv = static_cast<int>(v + 0.5f);
      if (iv > 127) iv = 127;
      if (iv < -128) iv = -128;
      dst[c * plane + i] = static_cast<int8_t>(iv);
    }
  }
  tensor->flushCache();
}

YoloXDetection::~YoloXDetection() {}
