#ifndef VPSS_PREPROCESSOR_H
#define VPSS_PREPROCESSOR_H

#include <cvi_comm_vpss.h>
#include <cstdint>

#include "preprocess/base_preprocessor.hpp"

class VpssContext {
 public:
  VpssContext();
  ~VpssContext();

  static VpssContext* GetInstance();

 private:
  static VpssContext instance_;
};
class VpssPreprocessor : public BasePreprocessor {
 public:
  VpssPreprocessor(int device = 0);
  ~VpssPreprocessor();

  std::shared_ptr<BaseImage> preprocess(
      const std::shared_ptr<BaseImage>& image, const PreprocessParams& params,
      std::shared_ptr<BaseMemoryPool> memory_pool = nullptr) override;
  int32_t preprocessToImage(const std::shared_ptr<BaseImage>& src_image,
                            const PreprocessParams& params,
                            std::shared_ptr<BaseImage> dst_image) override;
  int32_t preprocessToTensor(const std::shared_ptr<BaseImage>& src_image,
                             const PreprocessParams& params,
                             const int batch_idx,
                             std::shared_ptr<BaseTensor> tensor) override;

  void setUseVbPool(bool use_vb_pool) { use_vb_pool_ = use_vb_pool; }

  // Zero-copy support: when enabled, preprocessToTensor skips the CPU memcpy
  // (copyFromImage) in the stride-mismatch path and only populates
  // last_output_paddr_ with the physical address of the VPSS output buffer.
  // The caller must then redirect the TPU input tensor via
  // CviNet::setInputTensorPhysicalAddr before CVI_NN_Forward.
  // Always reset to false after each call to avoid accidental skips.
  void setZeroCopyHint(bool v) { zero_copy_hint_ = v; }
  uint64_t getLastOutputPaddr() const { return last_output_paddr_; }

 private:
  bool init();
  bool stop();
  int32_t prepareVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                            const PreprocessParams& params);
  int32_t generateVPSSGrpAttr(const std::shared_ptr<BaseImage>& src_image,
                              const PreprocessParams& params,
                              VPSS_GRP_ATTR_S& vpss_grp_attr) const;
  int32_t generateVPSSChnAttr(const std::shared_ptr<BaseImage>& src_image,
                              const PreprocessParams& params,
                              VPSS_CHN_ATTR_S& vpss_chn_attr) const;
  bool generateVPSSParams(const std::shared_ptr<BaseImage>& src_image,
                          const PreprocessParams& params,
                          VPSS_GRP_ATTR_S& vpss_grp_attr,
                          VPSS_CROP_INFO_S& vpss_chn_crop_attr,
                          VPSS_CHN_ATTR_S& vpss_chn_attr) const;
  int group_id_;
  int device_;
  VPSS_CROP_INFO_S crop_reset_attr_;
  bool use_vb_pool_ = false;

  // VPSS params cache — avoids redundant IOCTL calls when the source image
  // size/format and preprocessing params are unchanged between frames.
  uint32_t cached_src_w_     = 0;
  uint32_t cached_src_h_     = 0;
  int      cached_src_fmt_   = -1;
  uint32_t cached_dst_w_     = 0;
  uint32_t cached_dst_h_     = 0;
  int      cached_dst_fmt_   = -1;
  int      cached_dst_dtype_ = -1;
  bool     cached_nearest_   = false;
  bool     vpss_params_valid_ = false;  // false → must re-apply on next frame
  bool     zero_copy_hint_   = false;  // skip copyFromImage on next call

  // Keeps the last VPSS output image alive for zero-copy (see getLastOutputPaddr).
  std::shared_ptr<BaseImage> last_output_image_;
  uint64_t last_output_paddr_ = 0;
};

#endif  // VPSS_PREPROCESSOR_H