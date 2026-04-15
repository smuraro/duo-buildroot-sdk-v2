#include "vi_decoder/vi_decoder.hpp"
#include <cstdlib>
#include <mutex>
#include <queue>
#include <unistd.h>
#include <vector>
#include "image/base_image.hpp"
#include "memory/cvi_memory_pool.hpp"
#include "sample_comm.h"
#include "utils/tdl_log.hpp"

static SAMPLE_VI_CONFIG_S g_stViConfig = {};
std::vector<std::queue<std::shared_ptr<VIDEO_FRAME_INFO_S>>> frameQueues(
    VI_MAX_PIPE_NUM);
std::vector<std::mutex> queueMutexes(VI_MAX_PIPE_NUM);

// VPSS groups owned by the VI decoder across instances within this process.
// Used to perform a targeted cleanup on re-initialization without disturbing
// groups owned by the model VPSS preprocessor.
static std::vector<int32_t> g_vi_vpss_grps;

// ─── Persistent ISP subsystem ─────────────────────────────────────────────────
// On cv181x, vi_stop_streaming() sets the kernel-internal isp_streamoff flag to
// 1, but vi_start_streaming() never resets it.  Once set, the ISP SOF interrupt
// handler returns immediately, starving VPSS of frames on the next Camera open.
//
// Work-around (user-space only): keep VI/ISP alive for the lifetime of the
// process.  deinitialize() only tears down the VPSS group and VB pool; it never
// calls SAMPLE_COMM_VI_DestroyVI/ISP.  The final cleanup happens in the static
// destructor of ViSubsystemFinalizer, which runs at process exit.
//
// Side-effect: the sensor keeps streaming silently between Camera() open/close
// cycles within the same process.  This is intentional and acceptable for the
// single-process embedded use case.
// ─────────────────────────────────────────────────────────────────────────────
static std::mutex s_vi_mutex;
static int        s_refcount    = 0;   // active Camera() instances
static bool       s_isp_alive   = false;  // VI/ISP started by this process
static int32_t    s_init_w      = 0;
static int32_t    s_init_h      = 0;
static ImageFormat s_init_fmt   = ImageFormat::YUV420SP_VU;
static std::vector<int32_t>                       s_vpss_grps;
static std::vector<std::unique_ptr<MemoryBlock>>  s_memory_blocks;
static std::shared_ptr<BaseMemoryPool>            s_memory_pool;
static bool       s_sysinit_done = false;

// ─── Process-exit finalizer ───────────────────────────────────────────────────
// Explicit teardown of the VI/ISP/VPSS subsystem.
//
// Strategy: stop ISP first (no more SOF triggers → cvitask_vpss_1 goes idle),
// wait one frame period, then stop the VPSS group.  We deliberately skip
// CVI_VPSS_DisableChn (blocks on in-flight "work" jobs that need SOF to
// complete — hangs after ISP is stopped) and skip explicit VB/SYS teardown
// (the kernel driver releases those when the process file-descriptors close).
//
// This function is registered automatically as a C atexit handler the first
// time the camera subsystem is initialized (see vi_decoder_register_atexit).
// Python scripts that use the camera therefore exit cleanly with no changes.
//
// Can also be called explicitly from Python if needed:
//   import ctypes, os
//   ctypes.CDLL('libtdl_ex.so').vi_decoder_cleanup()
//   os._exit(0)
extern "C" void vi_decoder_cleanup() {
  std::lock_guard<std::mutex> sg(s_vi_mutex);
  if (!s_isp_alive) return;
#ifdef __CV184X__
  int32_t ViNum = g_stViConfig.s32ViNum;
#else
  int32_t ViNum = g_stViConfig.s32WorkingViNum;
#endif

  // Teardown order chosen to satisfy all three constraints:
  //
  //  (A) DisableChn must be called while ISP is still running.
  //      DisableChn blocks until the channel's "work" job queue drains.
  //      Jobs can only complete when ISP fires SOF events.  If ISP is stopped
  //      first, work jobs never complete and DisableChn hangs forever.
  //
  //  (B) StopGrp must precede UnBind.
  //      UnBind invalidates the ISP→VPSS VB pool link.  If cvitask_vpss_1 is
  //      mid-job calling _vb_dqbuf on that pool when UnBind happens, it
  //      crashes with a kernel use-after-free.  StopGrp calls jobs_exit()
  //      which cancels all in-flight jobs; afterwards cvitask_vpss_1 is idle
  //      and UnBind is safe.
  //
  //  (C) DestroyGrp must be called (not just StopGrp).
  //      The kernel VPSS driver does NOT destroy groups on fd close, so groups
  //      persist across processes.  Without DestroyGrp the next process's
  //      VpssPreprocessor (model inference) fails to get an available group.
  //
  // Order: StopGrp → DisableChn → DetachVbPool → UnBind → DestroyGrp
  //        → DestroyIsp → DestroyVi → _exit(0)

  // Step 1: stop group first (jobs_exit clears "work" queue, cvitask_vpss_1
  // goes idle) — must happen before UnBind to prevent _vb_dqbuf race
  for (int32_t i = 0; i < ViNum; i++) {
    int32_t grp = (i < (int32_t)s_vpss_grps.size()) ? s_vpss_grps[i] : i;
    CVI_VPSS_StopGrp(grp);
  }

  // Step 2: disable channel and drain done queue (ISP still running; since
  // jobs_exit already ran, work queue is empty → DisableChn returns quickly)
  for (int32_t i = 0; i < ViNum; i++) {
    int32_t grp = (i < (int32_t)s_vpss_grps.size()) ? s_vpss_grps[i] : i;
    CVI_VPSS_DisableChn(grp, VPSS_CHN0);
    VIDEO_FRAME_INFO_S f = {};
    while (CVI_VPSS_GetChnFrame(grp, VPSS_CHN0, &f, 0) == CVI_SUCCESS)
      CVI_VPSS_ReleaseChnFrame(grp, VPSS_CHN0, &f);
    CVI_VPSS_DetachVbPool(grp, VPSS_CHN0);
  }

  // Step 3: now that VPSS is idle, UnBind is safe (no active _vb_dqbuf calls)
  for (int32_t i = 0; i < ViNum; i++) {
    int32_t grp = (i < (int32_t)s_vpss_grps.size()) ? s_vpss_grps[i] : i;
    SAMPLE_COMM_VI_UnBind_VPSS(i, i, grp);
    CVI_VPSS_DestroyGrp(grp);
  }

  // Step 4: stop ISP/VI (vi_stop_streaming → isp_streamoff=1; irrelevant
  // since we are exiting and will never reinitialize in this process)
  SAMPLE_COMM_VI_DestroyIsp(&g_stViConfig);
  SAMPLE_COMM_VI_DestroyVi(&g_stViConfig);

  s_isp_alive = false;
  // VB blocks and SYS are intentionally left to the kernel driver to clean up
  // on fd close (os._exit or normal process termination).
}

// atexit callback: runs after Py_Finalize() but before C++ static destructors.
// Calls vi_decoder_cleanup() then _exit(0) to terminate immediately, bypassing
// the C++ static destructor below (which would run the same cleanup redundantly
// and risk a double-free or hang in a partially-shut-down environment).
static void vi_decoder_atexit_cb() {
  vi_decoder_cleanup();
  _exit(0);
}

// Registers the atexit callback exactly once (called from initialize()).
static void vi_decoder_register_atexit() {
  static std::once_flag s_flag;
  std::call_once(s_flag, []() { std::atexit(vi_decoder_atexit_cb); });
}

// Safety-net C++ destructor: runs only if _exit(0) was NOT called
// (e.g. the process was killed with SIGKILL or atexit was never registered
// because the camera was never opened).  vi_decoder_cleanup() is idempotent.
struct ViSubsystemFinalizer {
  ~ViSubsystemFinalizer() { vi_decoder_cleanup(); }
};
static ViSubsystemFinalizer s_vi_finalizer;

static PIXEL_FORMAT_E convertPixelFormat(ImageFormat img_format) {
  PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;

  if (img_format == ImageFormat::GRAY) {
    pixel_format = PIXEL_FORMAT_YUV_400;
  } else if (img_format == ImageFormat::YUV420SP_UV) {
    pixel_format = PIXEL_FORMAT_NV12;
  } else if (img_format == ImageFormat::YUV420SP_VU) {
    pixel_format = PIXEL_FORMAT_NV21;
  } else if (img_format == ImageFormat::YUV420P_UV) {
    LOGE("YUV420P_UV not support, imageFormat: %d", (int32_t)img_format);
  } else if (img_format == ImageFormat::YUV420P_VU) {
    LOGE("YUV420P_VU not support, imageFormat: %d", (int32_t)img_format);
  } else if (img_format == ImageFormat::RGB_PACKED) {
    pixel_format = PIXEL_FORMAT_RGB_888;
  } else if (img_format == ImageFormat::BGR_PACKED) {
    pixel_format = PIXEL_FORMAT_BGR_888;
  } else if (img_format == ImageFormat::RGB_PLANAR) {
    pixel_format = PIXEL_FORMAT_RGB_888_PLANAR;
  } else if (img_format == ImageFormat::BGR_PLANAR) {
    pixel_format = PIXEL_FORMAT_BGR_888_PLANAR;
  } else {
    LOGE("imageFormat not support, imageFormat: %d", (int32_t)img_format);
    pixel_format = PIXEL_FORMAT_MAX;
  }

  return pixel_format;
}

#ifdef __CV184X__
static int32_t TDL_PLAT_VI_INIT(SAMPLE_VI_CONFIG_S *pstViConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i = 0;

  /************************************************
   * Set sns reset, probe; Set MIPI attr
   ************************************************/
  SAMPLE_COMM_VI_StartMIPI(pstViConfig);

  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    if (!pstViConfig->astViInfo->stDevInfo.bPatgen) {
      if (CVI_SNS_SetSnsProbe(i) != CVI_SUCCESS) {
        LOGE("[ERROR] sensor_%d probe failed!\n", i);
        return CVI_FAILURE;
      }
    }
  }

  /************************************************
   * Set VI dev config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartDev(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartDev failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }
  /************************************************
   * Set VI pipe config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartPipe(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartPipe failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }

  /************************************************
   * Create ISP
   ************************************************/
  // to do
  s32Ret = SAMPLE_COMM_VI_CreateIsp(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("[ERROR] SAMPLE_COMM_VI_CreateIsp failed with %#x!\n", s32Ret);
    return s32Ret;
  }
  /************************************************
   * Set sensor init
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    if (!pstViConfig->astViInfo->stDevInfo.bPatgen) {
      if (CVI_SNS_SetSnsInit(i) != CVI_SUCCESS) {
        LOGE("[ERROR] sensor_%d init failed!\n", i);
        return CVI_FAILURE;
      }
    }
  }
  /************************************************
   * Set VI chn config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartChn(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartChn failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }

  return s32Ret;
}
#endif

// ─── VPSS-only init helper ───────────────────────────────────────────────────
// Creates VB pool + VPSS group + VI binding for one VI channel.
// Called both from the ISP-alive reopen path and the full init path.
static int32_t vpss_vb_init_for_channel(int32_t vi_ch, int32_t w, int32_t h,
                                         PIXEL_FORMAT_E pix_fmt,
                                         int32_t pool_id,
                                         uint32_t max_w, uint32_t max_h,
                                         int32_t *out_vpss_grp,
                                         bool mirror, bool flip) {
  VPSS_GRP VpssGrp = -1;
  {
    VPSS_GRP_ATTR_S probe_attr = {};
    for (VPSS_GRP g = 0; g <= 4; g++) {
      if (CVI_VPSS_GetGrpAttr(g, &probe_attr) != CVI_SUCCESS) {
        VpssGrp = g;
        break;
      }
    }
  }
  if (VpssGrp < 0) {
    LOGE("No available VPSS group in range 0-4 for ISP (online mode)\n");
    return -1;
  }
  LOGI("VI channel %d using VPSS group %d\n", vi_ch, VpssGrp);

  VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_CHN_ATTR_S astVpssChnAttr = {0};
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {CVI_FALSE};

  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_NV21;
  stVpssGrpAttr.u32MaxW = max_w;
  stVpssGrpAttr.u32MaxH = max_h;
#ifndef __CV186X__
  stVpssGrpAttr.u8VpssDev = 1;
#endif
  astVpssChnAttr.u32Width = w;
  astVpssChnAttr.u32Height = h;
  astVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
  astVpssChnAttr.enPixelFormat = pix_fmt;
  astVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
  astVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
  astVpssChnAttr.u32Depth = 1;
  astVpssChnAttr.bMirror = mirror ? CVI_TRUE : CVI_FALSE;
  astVpssChnAttr.bFlip   = flip  ? CVI_TRUE : CVI_FALSE;
  astVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
  astVpssChnAttr.stNormalize.bEnable = CVI_FALSE;
  abChnEnable[0] = CVI_TRUE;

  int32_t ret = 0;
#ifdef __CV184X__
  ret = SAMPLE_COMM_VPSS_INIT(VpssGrp, abChnEnable, &stVpssGrpAttr,
                              &astVpssChnAttr);
#else
  ret = SAMPLE_COMM_VPSS_Init(VpssGrp, abChnEnable, &stVpssGrpAttr,
                              &astVpssChnAttr);
#endif
  if (ret != CVI_SUCCESS) {
    LOGE("SAMPLE_COMM_VPSS_Init failed with ret: 0x%x !\n", ret);
    return ret;
  }

  ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, pool_id);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_AttachVbPool failed with ret: 0x%x !\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr,
                               &astVpssChnAttr);
  if (ret != CVI_SUCCESS) {
    LOGE("start vpss group failed. ret: 0x%x !\n", ret);
    return ret;
  }

  ret = SAMPLE_COMM_VI_Bind_VPSS(vi_ch, vi_ch, VpssGrp);
  if (ret != CVI_SUCCESS) {
    LOGE("vi bind vpss failed. ret: 0x%x !\n", ret);
    return ret;
  }

  *out_vpss_grp = VpssGrp;
  return CVI_SUCCESS;
}

int32_t ViDecoder::initialize(int32_t w, int32_t h, ImageFormat image_fmt,
                              int32_t vb_buffer_num, bool mirror, bool flip) {
  if (isInitialized) {
    LOGI("Camera have isInitialized\n");
    return 0;
  }

  LOGI("ViDecoder::initialize: s_isp_alive=%d s_refcount=%d req=%dx%d\n",
       (int)s_isp_alive, s_refcount, w, h);

  // ── Reuse path: VI/ISP/VPSS already alive (refcount may be 0) ────────────
  // Any VI unbind/rebind or VPSS stop/restart cycle triggers VIDIOC_STREAMOFF/
  // STREAMON on the VI device via CVI_SYS_UnBind/Bind.  This sets the kernel
  // isp_streamoff flag to 1 and vi_start_streaming never resets it, permanently
  // starving VPSS of frames.  The only safe user-space fix is to never call
  // UnBind/Bind or DestroyVI between Camera() open/close cycles: keep the whole
  // pipeline (VI, ISP, VPSS, binding) alive and stream silently between uses.
  // ViSubsystemFinalizer performs the full teardown at process exit.
  if (s_isp_alive) {
    if (w != s_init_w || h != s_init_h || image_fmt != s_init_fmt) {
      LOGE("ViDecoder: params mismatch with running subsystem "
           "(running %dx%d fmt=%d, requested %dx%d fmt=%d)\n",
           s_init_w, s_init_h, (int)s_init_fmt, w, h, (int)image_fmt);
      return -1;
    }
    vpss_grps_    = s_vpss_grps;
    memory_pool_  = s_memory_pool;
    {
      std::lock_guard<std::mutex> sg(s_vi_mutex);
      s_refcount++;
    }
    isInitialized  = true;
    sysinit_done_  = false;
    LOGI("ViDecoder: reusing running subsystem (refcount=%d)\n", s_refcount);
    return 0;
  }

  PIXEL_FORMAT_E pix_fmt = convertPixelFormat(image_fmt);
  int32_t ret = 0;

  // ── Full initialization (first Camera open in this process) ───────────────

  // Targeted cleanup: destroy VPSS groups previously allocated by this decoder
  // instance (tracked in g_vi_vpss_grps) plus any stale device-1 groups left by
  // a previous process that crashed without calling deinitialize().
  // VPSS groups are global kernel state and survive process death, so a crashed
  // session can leave device 1 (ISP) in a BUSY state that blocks the next run.
  // VpssPreprocessor (model) always uses device 0 (MEM), so it is safe to
  // destroy all device-1 groups unconditionally.
  {
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {CVI_FALSE};
    abChnEnable[0] = CVI_TRUE;

    if (!g_vi_vpss_grps.empty()) {
      for (int32_t i = 0; i < (int32_t)g_vi_vpss_grps.size(); i++) {
        int32_t grp = g_vi_vpss_grps[i];
        SAMPLE_COMM_VI_UnBind_VPSS(i, i, grp);
      }
    }
    if (!g_vi_vpss_grps.empty() || sysinit_done_) {
      SAMPLE_COMM_VI_DestroyIsp(&g_stViConfig);
      SAMPLE_COMM_VI_DestroyVi(&g_stViConfig);
    }
    for (int32_t i = 0; i < (int32_t)g_vi_vpss_grps.size(); i++) {
      int32_t grp = g_vi_vpss_grps[i];
      CVI_VPSS_StopGrp(grp);
      CVI_VPSS_DisableChn(grp, VPSS_CHN0);
      CVI_VPSS_DetachVbPool(grp, VPSS_CHN0);
      CVI_VPSS_DestroyGrp(grp);
    }
    if (!g_vi_vpss_grps.empty() || sysinit_done_) {
      if (sysinit_done_) {
        SAMPLE_COMM_SYS_Exit();
        sysinit_done_ = false;
      }
      g_vi_vpss_grps.clear();
      vpss_grps_.clear();
      memset(&g_stViConfig, 0, sizeof(SAMPLE_VI_CONFIG_S));
    }

    LOGI("Pre-init tracked cleanup done\n");
  }

#ifdef __CV184X__
  SNS_INI_CFG_S stIniCfg;
#else
  SAMPLE_INI_CFG_S stIniCfg;
#endif

  int32_t ViNum = 0;
  SAMPLE_VI_CONFIG_S stViConfig = {};
  VB_CONFIG_S stVbConf = {};
  PIC_SIZE_E enPicSize = {};
  VPSS_GRP VpssGrp = -1;
  int32_t pool_id[VI_MAX_PIPE_NUM] = {};

/************************************************
 * step1:  Config VI
 ************************************************/
#ifdef __CV184X__
  ret = CVI_SYS_Init();
  if (ret != 0) {
    LOGE("CVI_SYS_Init failed!\n");
    return ret;
  }

  ret = SAMPLE_COMM_VI_INI_INIT(&stViConfig, &stIniCfg, &stVbConf);
  if (ret != 0) {
    LOGE("SAMPLE_COMM_VI_INI_INIT fail\n");
    return ret;
  }
#else
  ret = SAMPLE_COMM_VI_ParseIni(&stIniCfg);
  if (ret != 0) {
    LOGE("Parse sensor_cfg.ini fail\n");
    return ret;
  }

  ret = SAMPLE_COMM_VI_IniToViCfg(&stIniCfg, &stViConfig);
  if (ret != 0) {
    LOGE("SAMPLE_COMM_VI_IniToViCfg fail\n");
    return ret;
  }
#endif

  CVI_VI_SetDevNum(stIniCfg.devNum);
  memcpy(&g_stViConfig, &stViConfig, sizeof(SAMPLE_VI_CONFIG_S));

  /************************************************
   * step2:  Initialize VB/SYS, then set VPSS/VI modes
   ************************************************/
  if (!CVI_VB_IsInited()) {
    memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt = 0;
    ret = SAMPLE_COMM_SYS_Init(&stVbConf);
    if (ret != 0) {
      LOGE("SAMPLE_COMM_SYS_Init failed. ret: 0x%x !\n", ret);
      return ret;
    }
    sysinit_done_  = true;
    s_sysinit_done = true;
  } else {
    LOGI("VB already initialized (model loaded first); skipping SAMPLE_COMM_SYS_Init\n");
    sysinit_done_  = false;
    s_sysinit_done = false;
  }

  VI_VPSS_MODE_S stVIVPSSMode;
  for (int i = 0; i < VI_MAX_PIPE_NUM; ++i) {
    stVIVPSSMode.aenMode[i] = VI_OFFLINE_VPSS_ONLINE;
  }
  CVI_SYS_SetVIVPSSMode(&stVIVPSSMode);

#ifndef __CV186X__
  VPSS_MODE_S stVPSSMode = {.enMode = VPSS_MODE_DUAL,
                            .aenInput = {VPSS_INPUT_MEM, VPSS_INPUT_ISP},
#ifndef __CV184X__
                            .ViPipe = {0}};
  CVI_SYS_SetVPSSModeEx(&stVPSSMode);
#else
                           };
  CVI_VPSS_SetMode(&stVPSSMode);
#endif
#endif

  // Stale device-1 cleanup
#ifndef __CV186X__
  for (VPSS_GRP scan_grp = 0; scan_grp < VPSS_MAX_GRP_NUM; scan_grp++) {
    VPSS_GRP_ATTR_S scan_attr = {};
    if (CVI_VPSS_GetGrpAttr(scan_grp, &scan_attr) != CVI_SUCCESS) continue;
    if (scan_attr.u8VpssDev != 1) continue;
    LOGI("Stale device-1 VPSS group %d found; cleaning up\n", scan_grp);
    CVI_VPSS_StopGrp(scan_grp);
    CVI_VPSS_DisableChn(scan_grp, VPSS_CHN0);
    CVI_VPSS_DestroyGrp(scan_grp);
  }
#endif

  memory_pool_ = MemoryPoolFactory::createMemoryPool();
  memory_blocks_.clear();
  auto pool = std::dynamic_pointer_cast<CviMemoryPool>(memory_pool_);
  if (!pool) {
    LOGE("memory_pool is nullptr");
    return -1;
  }

#ifdef __CV184X__
  ViNum = stViConfig.s32ViNum;
#else
  ViNum = stViConfig.s32WorkingViNum;
#endif

  SIZE_S stSize[ViNum] = {};
  for (int32_t i = 0; i < ViNum; i++) {
    ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType[i], &enPicSize);
    if (ret != CVI_SUCCESS) {
      LOGE("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", ret);
      return ret;
    }
    ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize[i]);
    if (ret != CVI_SUCCESS) {
      LOGE("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", ret);
      return ret;
    }
    memory_blocks_.push_back(
        pool->CreateExVb(vb_buffer_num, w, h, (void *)&pix_fmt));
    pool_id[i] = memory_blocks_.back()->id;
  }

  /************************************************
   * step3:  Init vi modules
   ************************************************/
#ifdef __CV184X__
  ret = TDL_PLAT_VI_INIT(&stViConfig);
  if (ret != 0) {
    LOGE("TDL_PLAT_VI_INIT failed. ret: 0x%x !\n", ret);
    return ret;
  }
#else
  ret = SAMPLE_PLAT_VI_INIT(&stViConfig);
  if (ret != 0) {
    LOGE("SAMPLE_PLAT_VI_INIT failed. ret: 0x%x !\n", ret);
    return ret;
  }
#endif

  s_isp_alive = true;
  vi_decoder_register_atexit();  // auto-cleanup on process exit

  /************************************************
   * step4:  Init vpss modules
   ************************************************/
  for (int32_t i = 0; i < ViNum; i++) {
    int32_t vpss_grp = -1;
    ret = vpss_vb_init_for_channel(i, w, h, pix_fmt, pool_id[i],
                                   stSize[i].u32Width, stSize[i].u32Height,
                                   &vpss_grp, mirror, flip);
    if (ret != CVI_SUCCESS) return ret;
    VpssGrp = vpss_grp;
    vpss_grps_.push_back(VpssGrp);
    g_vi_vpss_grps.push_back(VpssGrp);
  }

  // ── Commit to singleton state ───────────────────────────────────────────────
  {
    std::lock_guard<std::mutex> sg(s_vi_mutex);
    s_init_w        = w;
    s_init_h        = h;
    s_init_fmt      = image_fmt;
    s_vpss_grps     = vpss_grps_;
    s_memory_pool   = memory_pool_;
    s_memory_blocks = std::move(memory_blocks_);
    s_refcount      = 1;
  }

  isInitialized = true;
  return ret;
}

int32_t ViDecoder::deinitialize() {
  int32_t ret = 0;
  int32_t ViNum = 0;

#ifdef __CV184X__
  ViNum = g_stViConfig.s32ViNum;
#else
  ViNum = g_stViConfig.s32WorkingViNum;
#endif

  // Drain frames held by this instance — but only if the VPSS groups are still
  // alive.  If vi_decoder_cleanup() has already been called (s_isp_alive=false)
  // the VPSS groups are destroyed and CVI_VPSS_ReleaseChnFrame would fail with
  // "Grp not yet started"; skip the drain in that case and just clear the queue.
  for (int32_t i = 0; i < ViNum; i++) {
    int32_t grp = (i < (int32_t)vpss_grps_.size()) ? vpss_grps_[i] : i;
    std::unique_lock<std::mutex> lock(queueMutexes[i]);
    while (!frameQueues[i].empty()) {
      auto frame_info = frameQueues[i].front();
      frameQueues[i].pop();
      for (int32_t j = 0; j < 3; j++) {
        if (frame_info->stVFrame.u32Length[j] != 0) {
          CVI_SYS_Munmap((void *)frame_info->stVFrame.pu8VirAddr[j],
                         frame_info->stVFrame.u32Length[j]);
        }
      }
      if (s_isp_alive) {
        CVI_VPSS_ReleaseChnFrame(grp, VPSS_CHN0, frame_info.get());
      }
    }
  }

  LOGI("ViDecoder::deinitialize: s_isp_alive=%d s_refcount=%d\n",
       (int)s_isp_alive, s_refcount);

  // ── Keep-alive close: VI/ISP/VPSS/binding all remain running ─────────────
  // Do NOT call UnBind, StopGrp, DestroyGrp, or DestroyVI here.
  // Any of those operations eventually issues VIDIOC_STREAMOFF/STREAMON to the
  // VI V4L2 device, setting isp_streamoff=1 without ever resetting it, which
  // permanently blocks the ISP SOF handler from triggering VPSS on the next open.
  // The pipeline streams silently (sensor → ISP → VPSS doneq accumulates frames)
  // until the next Camera() open drains and resumes reading.
  // ViSubsystemFinalizer handles the full teardown at process exit.
  {
    std::lock_guard<std::mutex> sg(s_vi_mutex);
    s_refcount--;
    LOGI("ViDecoder: keep-alive close (refcount=%d, pipeline stays running)\n",
         s_refcount);
  }
  vpss_grps_.clear();
  memory_pool_.reset();
  isInitialized = false;
  sysinit_done_ = false;
  return ret;
}

ViDecoder::ViDecoder() {
  type_ = VideoDecoderType::VI;
}

ViDecoder::~ViDecoder() {
  if (isInitialized || sysinit_done_) {
    deinitialize();
  }
}

int32_t ViDecoder::init(const std::string &path,
                        const std::map<std::string, int32_t> &config) {
  path_ = path;
  return 0;
}

int32_t ViDecoder::read(std::shared_ptr<BaseImage> &image, int32_t vi_chn) {
  int32_t ret = 0;
  int32_t vpss_grp = (vi_chn < (int32_t)vpss_grps_.size()) ? vpss_grps_[vi_chn] : vi_chn;
  std::shared_ptr<VIDEO_FRAME_INFO_S> frame_info =
      std::make_shared<VIDEO_FRAME_INFO_S>();
  while (true) {
    ret = CVI_VPSS_GetChnFrame(vpss_grp, 0, frame_info.get(), 3000);
    VIDEO_FRAME_INFO_S *vpss_frame_info = frame_info.get();
    if (ret != 0 ||
        vpss_frame_info->stVFrame.u32Width == 0) {  // CVI_VPSS_GetChnFrame bug
      LOGE("CVI_VPSS_GetChnFrame(grp:%d) failed, ret(%x) width %d\n", vpss_grp,
           ret, vpss_frame_info->stVFrame.u32Width);
      return ret;
    } else {
      break;
    }
  }

  if (frame_info->stVFrame.pu8VirAddr[0] == NULL) {
    isMapped_ = true;
    for (int32_t i = 0; i < 3; i++) {
      if (frame_info->stVFrame.u32Length[i] != 0) {
        frame_info->stVFrame.pu8VirAddr[i] =
            (CVI_U8 *)CVI_SYS_Mmap(frame_info->stVFrame.u64PhyAddr[i],
                                   frame_info->stVFrame.u32Length[i]);
        CVI_SYS_IonFlushCache(frame_info->stVFrame.u64PhyAddr[i],
                              frame_info->stVFrame.pu8VirAddr[i],
                              frame_info->stVFrame.u32Length[i]);
        addr_[i] = frame_info->stVFrame.pu8VirAddr[i];
        image_length_[i] = frame_info->stVFrame.u32Length[i];
      }
    }
  }

  image = ImageFactory::wrapVPSSFrame(frame_info.get(), false);

  std::lock_guard<std::mutex> lock(queueMutexes[vi_chn]);
  frameQueues[vi_chn].push(frame_info);
  frame_id_++;

  return image ? 0 : -1;
}

int32_t ViDecoder::release(int32_t vi_chn) {
  int32_t ret = 0;
  int32_t vpss_grp = (vi_chn < (int32_t)vpss_grps_.size()) ? vpss_grps_[vi_chn] : vi_chn;
  if (frameQueues[vi_chn].empty()) {
    LOGE("FrameBuffer is empty\n");
    return -1;
  }

  std::unique_lock<std::mutex> lock(queueMutexes[vi_chn]);
  auto frame_info = frameQueues[vi_chn].front();
  frameQueues[vi_chn].pop();

  for (int32_t i = 0; i < 3; i++) {
    if (frame_info->stVFrame.u32Length[i] != 0) {
      CVI_SYS_Munmap((void *)frame_info->stVFrame.pu8VirAddr[i],
                     frame_info->stVFrame.u32Length[i]);
    }
  }

  ret = CVI_VPSS_ReleaseChnFrame(vpss_grp, 0, frame_info.get());
  if (ret != 0) {
    LOGE("CVI_VPSS_ReleaseChnFrame(grp:%d) failed with %d\n", vpss_grp, ret);
  }
  return ret;
}
