// py_rtsp_client_vdec.cpp
//
// Hardware-accelerated RTSP client:
//   live555 (RTSP/RTP)  →  CVITEK VDEC (H264 hardware decode)
//   →  VIDEO_FRAME_INFO_S (YUV420 NV12 in VB memory)
//   →  VPSSImage  →  PyImage
//
// H265 streams use an OpenCV/FFmpeg software fallback (no hardware support
// for H265 decode on CV181X in practice).
//
// Thread model
// ─────────────
//   event_thread_  : runs the live555 scheduler (doEventLoop).  Handles all
//                    RTSP negotiation (DESCRIBE/SETUP/PLAY) and RTP receive.
//                    Sends compressed NAL units to the VDEC driver via
//                    CVI_VDEC_SendStream.
//
//   Python thread  : calls read() / release() which call CVI_VDEC_GetFrame /
//                    CVI_VDEC_ReleaseFrame.  The VDEC kernel driver is
//                    internally thread-safe.
//
// Initialization handshake
// ─────────────────────────
//   The constructor waits on ready_future_.  The event thread fulfils the
//   promise once the first frame arrives at VdecSink (so VDEC is confirmed
//   to be receiving data) — or immediately on any setup failure.
//
// CVI_VDEC_GetFrame semantics
// ────────────────────────────
//   The kernel implementation is NOT a blocking wait.  It checks the display
//   queue and returns CVI_ERR_VDEC_ERR_INVALID_RET (0xc0058041) immediately
//   when empty.  read() therefore polls with a 10 ms interval up to timeout_ms_.

#include "py_rtsp_client_vdec.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// ── live555 ──────────────────────────────────────────────────────────────────
#include <BasicUsageEnvironment.hh>
#include <liveMedia.hh>

// ── CVI middleware ────────────────────────────────────────────────────────────
#include "cvi_buffer.h"
#include "cvi_sys.h"
#include "cvi_vb.h"

// ── TDL SDK ──────────────────────────────────────────────────────────────────
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"

namespace pytdl {

std::atomic<int> PyRtspClientVdec::next_vdec_chn_{0};

// ─────────────────────────────────────────────────────────────────────────────
// Anonymous namespace — live555 helper classes and static callbacks
// ─────────────────────────────────────────────────────────────────────────────
namespace {

static constexpr size_t kRtpBufSize = 512 * 1024;  // 512 KB per NAL unit

// Forward declarations for the callback chain
static void continueAfterDESCRIBE(RTSPClient*, int, char*);
static void continueAfterSETUP(RTSPClient*, int, char*);
static void continueAfterPLAY(RTSPClient*, int, char*);
static void subsessionAfterPlaying(void*);

// ── Per-session state ────────────────────────────────────────────────────────
struct StreamState {
  MediaSubsessionIterator* iter       = nullptr;
  MediaSession*            session    = nullptr;
  MediaSubsession*         subsession = nullptr;
};

// ── TdlRTSPClient ─────────────────────────────────────────────────────────────
class TdlRTSPClient : public RTSPClient {
 public:
  static TdlRTSPClient* createNew(UsageEnvironment& env,
                                   const char* url,
                                   PyRtspClientVdec* parent,
                                   int vdec_chn,
                                   int target_w, int target_h,
                                   bool use_tcp,
                                   char& event_watch) {
    return new TdlRTSPClient(env, url, parent, vdec_chn,
                              target_w, target_h, use_tcp, event_watch);
  }

  PyRtspClientVdec* parent;
  int   vdec_chn;
  int   target_w;
  int   target_h;
  bool  use_tcp;
  char& event_watch;

  bool     promise_fulfilled = false;
  bool     is_h265           = false;
  uint32_t stream_w          = 0;
  uint32_t stream_h          = 0;

  StreamState scs;

 protected:
  TdlRTSPClient(UsageEnvironment& env, const char* url,
                 PyRtspClientVdec* p, int vc, int tw, int th,
                 bool tcp, char& ew)
      : RTSPClient(env, url, 0, "TdlRtspClientVdec", 0, -1),
        parent(p), vdec_chn(vc), target_w(tw), target_h(th),
        use_tcp(tcp), event_watch(ew) {}

  ~TdlRTSPClient() override {
    delete scs.iter;
    if (scs.session) Medium::close(scs.session);
  }
};

// ── VdecSink ──────────────────────────────────────────────────────────────────
// Receives reassembled NAL units from live555 and feeds them to VDEC.
class VdecSink : public MediaSink {
 public:
  static VdecSink* createNew(UsageEnvironment& env,
                              MediaSubsession& subsession,
                              TdlRTSPClient* client) {
    return new VdecSink(env, subsession, client);
  }

 private:
  VdecSink(UsageEnvironment& env, MediaSubsession& sub,
           TdlRTSPClient* client)
      : MediaSink(env),
        client_(client),
        vdec_chn_(client->vdec_chn) {
    // Receive buffer: start code (4 B) + NAL payload
    buf_ = new uint8_t[kRtpBufSize + 4];
    buf_[0] = 0; buf_[1] = 0; buf_[2] = 0; buf_[3] = 1;

    // H264: inject SDP SPS/PPS so the Coda9 decoder has parameter sets before
    // the first frame arrives.
    const char* codec = sub.codecName();
    if (codec && strcmp(codec, "H264") == 0)
      injectSPropSet(sub.fmtp_spropparametersets());
  }

  ~VdecSink() override { delete[] buf_; }

  // Collect NAL units from a base64-encoded sprop string into dst (Annex-B).
  static void collectSPropSet(const char* sprop, std::vector<uint8_t>& dst) {
    if (!sprop || sprop[0] == '\0') return;
    unsigned n = 0;
    SPropRecord* recs = parseSPropParameterSets(sprop, n);
    for (unsigned i = 0; i < n; ++i) {
      if (recs[i].sPropBytes && recs[i].sPropLength > 0) {
        dst.push_back(0); dst.push_back(0);
        dst.push_back(0); dst.push_back(1);
        dst.insert(dst.end(),
                   recs[i].sPropBytes,
                   recs[i].sPropBytes + recs[i].sPropLength);
      }
    }
    delete[] recs;
  }

  void injectSPropSet(const char* sprop) {
    std::vector<uint8_t> buf;
    collectSPropSet(sprop, buf);
    if (buf.empty()) return;
    VDEC_STREAM_S s{};
    s.pu8Addr      = buf.data();
    s.u32Len       = static_cast<CVI_U32>(buf.size());
    s.bEndOfFrame  = CVI_TRUE;
    s.bEndOfStream = CVI_FALSE;
    CVI_VDEC_SendStream(vdec_chn_, &s, 2000);
  }

  // ── MediaSink interface ──────────────────────────────────────────────────
  Boolean continuePlaying() override {
    if (!fSource) return False;
    fSource->getNextFrame(buf_ + 4, kRtpBufSize,
                          afterGettingFrame, this,
                          onSourceClosure, this);
    return True;
  }

  static void afterGettingFrame(void* self, unsigned size, unsigned /*trunc*/,
                                 struct timeval pts, unsigned /*dur*/) {
    static_cast<VdecSink*>(self)->onFrame(size, pts);
  }

  void onFrame(unsigned size, struct timeval pts) {
    if (size > 0) {
      uint32_t nal_len = 4 + static_cast<uint32_t>(size);
      CVI_U64  pts_us  = static_cast<CVI_U64>(pts.tv_sec) * 1000000ULL
                       + static_cast<CVI_U64>(pts.tv_usec);

      // H264: each RTP NAL unit is one complete access unit.
      VDEC_STREAM_S s{};
      s.pu8Addr      = buf_;
      s.u32Len       = nal_len;
      s.bEndOfFrame  = CVI_TRUE;
      s.bEndOfStream = CVI_FALSE;
      s.bDisplay     = CVI_TRUE;
      s.u64PTS       = pts_us;
      CVI_VDEC_SendStream(vdec_chn_, &s, 2000);
    }

    // Notify the constructor that data is flowing (once only).
    if (!client_->promise_fulfilled) {
      client_->promise_fulfilled = true;
      client_->parent->onVdecReady(true);
    }

    continuePlaying();
  }

  TdlRTSPClient* client_;
  int            vdec_chn_;
  uint8_t*       buf_;
};

// ── subsessionAfterPlaying ────────────────────────────────────────────────────
static void subsessionAfterPlaying(void* clientData) {
  auto* sub = static_cast<MediaSubsession*>(clientData);
  Medium::close(sub->sink);
  sub->sink = nullptr;
}

// ── continueAfterPLAY ────────────────────────────────────────────────────────
static void continueAfterPLAY(RTSPClient* rtspClient, int resultCode,
                               char* resultString) {
  auto* client = static_cast<TdlRTSPClient*>(rtspClient);
  delete[] resultString;

  if (resultCode != 0) {
    LOGE("[VdecRtsp] PLAY failed: %d\n", resultCode);
    if (!client->promise_fulfilled) {
      client->promise_fulfilled = true;
      client->parent->onVdecReady(false);
    }
    client->event_watch = 1;
    return;
  }

  // Attach VdecSink to each video subsession and start playing.
  MediaSubsessionIterator iter(*client->scs.session);
  MediaSubsession* sub;
  while ((sub = iter.next()) != nullptr) {
    if (!sub->sink) continue;
    sub->miscPtr = client;
    sub->sink->startPlaying(*sub->readSource(),
                             subsessionAfterPlaying, sub);
  }
  LOGI("[VdecRtsp] PLAY started\n");
}

// ── continueAfterSETUP ───────────────────────────────────────────────────────
static void continueAfterSETUP(RTSPClient* rtspClient, int resultCode,
                                char* resultString) {
  auto* client = static_cast<TdlRTSPClient*>(rtspClient);
  UsageEnvironment& env = rtspClient->envir();
  delete[] resultString;

  if (resultCode == 0) {
    // SETUP succeeded — create a VdecSink for this subsession.
    MediaSubsession* sub = client->scs.subsession;
    sub->sink = VdecSink::createNew(env, *sub, client);
    if (!sub->sink)
      LOGE("[VdecRtsp] VdecSink creation failed\n");
  }

  // Advance to the next video subsession that still needs SETUP.
  MediaSubsession* next = nullptr;
  while ((next = client->scs.iter->next()) != nullptr) {
    if (strcmp(next->mediumName(), "video") == 0) break;
  }

  if (next) {
    client->scs.subsession = next;
    next->initiate();
    rtspClient->sendSetupCommand(*next, continueAfterSETUP,
                                  False,              // streamOutgoing
                                  client->use_tcp);   // streamUsingTCP
    return;
  }

  // All done — send PLAY.
  rtspClient->sendPlayCommand(*client->scs.session, continueAfterPLAY);
}

// ── continueAfterDESCRIBE ────────────────────────────────────────────────────
static void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode,
                                   char* resultString) {
  auto* client = static_cast<TdlRTSPClient*>(rtspClient);

  auto fail = [&](const char* msg) {
    LOGE("[VdecRtsp] DESCRIBE/setup error: %s\n", msg);
    delete[] resultString;
    if (!client->promise_fulfilled) {
      client->promise_fulfilled = true;
      client->parent->onVdecReady(false);
    }
    client->event_watch = 1;
  };

  if (resultCode != 0) {
    fail(resultString ? resultString : "(no detail)");
    return;
  }

  UsageEnvironment& env = rtspClient->envir();

  // resultString is the SDP description.
  MediaSession* session = MediaSession::createNew(env, resultString);
  delete[] resultString;
  resultString = nullptr;

  if (!session || !session->hasSubsessions()) {
    fail("empty or unparseable SDP");
    if (session) Medium::close(session);
    return;
  }
  client->scs.session = session;

  // Find the first video subsession to determine codec and resolution.
  client->scs.iter = new MediaSubsessionIterator(*session);
  MediaSubsession* firstVideo = nullptr;
  while ((firstVideo = client->scs.iter->next()) != nullptr) {
    if (strcmp(firstVideo->mediumName(), "video") == 0) break;
  }
  if (!firstVideo) {
    fail("no video subsession");
    return;
  }

  // Codec detection
  const char* codec = firstVideo->codecName();
  client->is_h265 = codec && strcmp(codec, "H265") == 0;

  // Stream resolution from SDP (may be 0 for some cameras)
  uint32_t sdp_w = firstVideo->videoWidth();
  uint32_t sdp_h = firstVideo->videoHeight();
  client->stream_w = sdp_w > 0  ? sdp_w
                  : (client->target_w > 0 ? static_cast<uint32_t>(client->target_w)
                                          : 1920u);
  client->stream_h = sdp_h > 0  ? sdp_h
                  : (client->target_h > 0 ? static_cast<uint32_t>(client->target_h)
                                          : 1080u);

  LOGI("[VdecRtsp] stream: codec=%s  %ux%u\n",
       codec ? codec : "?", client->stream_w, client->stream_h);

  // H265: CV181X Wave4 has no working H265 hardware decode — use OpenCV fallback.
  if (client->is_h265) {
    LOGW("[VdecRtsp] H265 não tem suporte de hardware no CV181X — "
         "usando fallback OpenCV/FFmpeg\n");
    bool ok = client->parent->openCvFallback();
    client->promise_fulfilled = true;
    client->parent->onVdecReady(ok);
    client->event_watch = 1;  // stop live555 loop — we don't need it
    return;
  }

  // Initialise the VDEC hardware channel now that we know the codec/resolution.
  if (!client->parent->initVdec(client->stream_w, client->stream_h,
                                 /*is_h265=*/false)) {
    fail("VDEC init failed");
    return;
  }

  // Reset iterator, find first video subsession, and begin SETUP chain.
  delete client->scs.iter;
  client->scs.iter = new MediaSubsessionIterator(*session);
  MediaSubsession* sub = nullptr;
  while ((sub = client->scs.iter->next()) != nullptr) {
    if (strcmp(sub->mediumName(), "video") == 0) break;
  }
  if (!sub) {
    fail("no video subsession (2nd pass)");
    return;
  }

  client->scs.subsession = sub;
  sub->initiate();
  rtspClient->sendSetupCommand(*sub, continueAfterSETUP,
                                False,             // streamOutgoing
                                client->use_tcp);  // streamUsingTCP
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// PyRtspClientVdec — public implementation
// ─────────────────────────────────────────────────────────────────────────────

PyRtspClientVdec::PyRtspClientVdec(const std::string& url,
                                   int width, int height,
                                   int timeout_ms,
                                   const std::string& transport)
    : url_(url),
      transport_(transport),
      timeout_ms_(timeout_ms),
      target_width_(width),
      target_height_(height),
      ready_future_(ready_promise_.get_future()) {
  vdec_chn_ = next_vdec_chn_.fetch_add(1);

  LOGI("[VdecRtsp] opening %s  target=%dx%d  vdec_chn=%d  transport=%s\n",
       url.c_str(), width, height, vdec_chn_, transport.c_str());

  event_thread_ = std::thread(&PyRtspClientVdec::eventLoopThread, this);

  // Block until RTSP negotiation + VDEC init + first frame confirm.
  bool ok = ready_future_.get();
  if (!ok) {
    event_watch_ = 1;
    if (event_thread_.joinable()) event_thread_.join();
    deinitVdec();
    throw std::runtime_error(
        "RtspClientVdec: failed to open stream: " + url);
  }
  is_opened_ = true;
  LOGI("[VdecRtsp] ready on VDEC channel %d\n", vdec_chn_);
}

PyRtspClientVdec::~PyRtspClientVdec() { close(); }

void PyRtspClientVdec::eventLoopThread() {
  TaskScheduler*    scheduler = BasicTaskScheduler::createNew();
  UsageEnvironment* env       = BasicUsageEnvironment::createNew(*scheduler);

  bool use_tcp = (transport_ == "tcp");

  TdlRTSPClient* client = TdlRTSPClient::createNew(
      *env, url_.c_str(),
      this, vdec_chn_,
      target_width_, target_height_,
      use_tcp, event_watch_);

  if (!client) {
    LOGE("[VdecRtsp] failed to create TdlRTSPClient\n");
    try { ready_promise_.set_value(false); } catch (...) {}
    env->reclaim();
    delete scheduler;
    return;
  }

  client->sendDescribeCommand(continueAfterDESCRIBE);

  // Blocks until event_watch_ != 0
  scheduler->doEventLoop(&event_watch_);

  Medium::close(client);
  env->reclaim();
  delete scheduler;
}

void PyRtspClientVdec::onVdecReady(bool ok) {
  try { ready_promise_.set_value(ok); } catch (...) {}
}

bool PyRtspClientVdec::openCvFallback() {
  use_cv_fallback_ = true;
  cv_cap_.open(url_, cv::CAP_FFMPEG);
  if (!cv_cap_.isOpened()) {
    LOGE("[VdecRtsp] H265 fallback: falha ao abrir stream com OpenCV: %s\n",
         url_.c_str());
    return false;
  }
  if (timeout_ms_ > 0) {
    cv_cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms_);
    cv_cap_.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms_);
  }
  LOGI("[VdecRtsp] H265 fallback OpenCV aberto: %s\n", url_.c_str());
  return true;
}

bool PyRtspClientVdec::initVdec(uint32_t width, uint32_t height, bool is_h265) {
  PAYLOAD_TYPE_E codec = is_h265 ? PT_H265 : PT_H264;

  // Ensure CVI_SYS and CVI_VB are initialised.  When running without Camera,
  // neither is initialised and CVI_VB_CreatePool / CVI_VDEC_AttachVbPool will
  // silently fail.  CVI_SYS_Init / CVI_VB_Init are idempotent — calling them
  // again when already initialised is safe (they return CVI_SUCCESS).
  {
    VB_CONFIG_S vb_cfg{};
    vb_cfg.u32MaxPoolCnt = 2;  // pic pool + tmv pool
    // Block sizes/counts will be set via CVI_VB_CreatePool; leave the
    // static pools at zero so VB_Init just brings up the subsystem.
    int r = CVI_VB_SetConfig(&vb_cfg);
    if (r != 0) LOGIP("[VdecRtsp] CVI_VB_SetConfig ret=0x%x (ignored)\n", r);
    r = CVI_VB_Init();
    if (r != 0) LOGIP("[VdecRtsp] CVI_VB_Init ret=0x%x (ignored)\n", r);
    r = CVI_SYS_Init();
    if (r != 0) LOGIP("[VdecRtsp] CVI_SYS_Init ret=0x%x (ignored)\n", r);
  }

  // VDEC max size = max(target, stream); align to 16-pixel boundary.
  uint32_t max_w = std::max({(uint32_t)target_width_, width, 1u});
  uint32_t max_h = std::max({(uint32_t)target_height_, height, 1u});
  max_w = (max_w + 15u) & ~15u;
  max_h = (max_h + 15u) & ~15u;

  // Stream buffer: ALIGN(width * height, 16KB).
  uint32_t stream_buf_size = (max_w * max_h + 0x3FFFu) & ~0x3FFFu;
  if (stream_buf_size < 1u * 1024u * 1024u)
    stream_buf_size = 1u * 1024u * 1024u;

  // H265 needs more frame buffers for the DPB (Wave4 spec).
  // 20 blocks: num_ref_frames(up to 16) + displayFrameNum(2) + margin.
  // 20 blocks: num_ref_frames (up to 16 for H264 High Profile) +
  // displayFrameNum(2) + margin.  Applies to both codecs.
  const uint32_t kFrameBufCnt = 20u;

  // ── VB_SOURCE_USER: create dedicated VB pools ─────────────────────────────
  // The common VB pool is not initialised when running without Camera.
  // Wave4 (H265) requires a TMV (co-located motion vector) pool in addition
  // to the picture pool; without it the VCPU crashes during seq-init with
  // W4_RST_BLOCK_VCPU (0x1000000).
  //
  // Picture pool block size: use VDEC_GetPicBufferSize so alignment is correct.
  uint32_t pic_blk_size = VDEC_GetPicBufferSize(codec, max_w, max_h,
                                                  PIXEL_FORMAT_NV12,
                                                  DATA_BITWIDTH_8,
                                                  COMPRESS_MODE_NONE);

  // TMV pool block size: (w/16) * (h/16) * 32 bytes, based on ACTUAL stream
  // dimensions (not max_w/max_h) as the Wave4 validates the TMV size against
  // the SPS dimensions.
  uint32_t tmv_w = (width  + 15u) & ~15u;
  uint32_t tmv_h = (height + 15u) & ~15u;
  uint32_t tmv_blk_size = ((tmv_w + 63u) / 64u) * ((tmv_h + 63u) / 64u) * 512u;
  if (tmv_blk_size < 4096u) tmv_blk_size = 4096u;


  // Create picture VB pool (kFrameBufCnt blocks).
  {
    VB_POOL_CONFIG_S cfg{};
    cfg.u32BlkSize  = pic_blk_size;
    cfg.u32BlkCnt   = kFrameBufCnt;
    cfg.enRemapMode = VB_REMAP_MODE_NOCACHE;
    vb_pool_ = CVI_VB_CreatePool(&cfg);
    if (vb_pool_ == VB_INVALID_POOLID) {
      LOGE("[VdecRtsp] CVI_VB_CreatePool (pic) failed\n");
      return false;
    }
  }

  // Create TMV pool (H265 only, kFrameBufCnt blocks).
  if (is_h265) {
    VB_POOL_CONFIG_S cfg{};
    cfg.u32BlkSize  = tmv_blk_size;
    cfg.u32BlkCnt   = kFrameBufCnt;
    cfg.enRemapMode = VB_REMAP_MODE_NOCACHE;
    vb_tmv_pool_ = CVI_VB_CreatePool(&cfg);
    if (vb_tmv_pool_ == VB_INVALID_POOLID) {
      LOGE("[VdecRtsp] CVI_VB_CreatePool (tmv) failed\n");
      CVI_VB_DestroyPool(vb_pool_);
      vb_pool_ = VB_INVALID_POOLID;
      return false;
    }
  }

  // Switch VDEC to VB_SOURCE_USER so CVI_VDEC_AttachVbPool is accepted.
  {
    VDEC_MOD_PARAM_S mod_param{};
    CVI_VDEC_GetModParam(&mod_param);
    mod_param.enVdecVBSource = VB_SOURCE_USER;
    CVI_VDEC_SetModParam(&mod_param);
  }

  VDEC_CHN_ATTR_S attr{};
  attr.enType           = codec;
  attr.enMode           = VIDEO_MODE_FRAME;
  attr.u32PicWidth      = max_w;
  attr.u32PicHeight     = max_h;
  attr.u32StreamBufSize = stream_buf_size;
  attr.u32FrameBufSize  = pic_blk_size;
  attr.u32FrameBufCnt   = kFrameBufCnt;

  int ret = CVI_VDEC_CreateChn(vdec_chn_, &attr);
  if (ret != 0) {
    LOGE("[VdecRtsp] CVI_VDEC_CreateChn(%d) ret=0x%x\n", vdec_chn_, ret);
    goto err_pools;
  }

  // Attach the dedicated VB pools to the channel.
  {
    VDEC_CHN_POOL_S pool{};
    pool.hPicVbPool = vb_pool_;
    pool.hTmvVbPool = is_h265 ? vb_tmv_pool_ : VB_INVALID_POOLID;
    ret = CVI_VDEC_AttachVbPool(vdec_chn_, &pool);
    if (ret != 0) {
      LOGE("[VdecRtsp] CVI_VDEC_AttachVbPool ret=0x%x\n", ret);
      CVI_VDEC_DestroyChn(vdec_chn_);
      goto err_pools;
    }
  }

  {
    VDEC_CHN_PARAM_S param{};
    param.enType             = codec;
    param.enPixelFormat      = PIXEL_FORMAT_NV12;
    param.u32DisplayFrameNum = 4;
    param.stVdecVideoParam.enDecMode      = VIDEO_DEC_MODE_IPB;
    param.stVdecVideoParam.enOutputOrder  = VIDEO_OUTPUT_ORDER_DISP;
    param.stVdecVideoParam.enCompressMode = COMPRESS_MODE_NONE;

    ret = CVI_VDEC_SetChnParam(vdec_chn_, &param);
    if (ret != 0) {
      LOGE("[VdecRtsp] CVI_VDEC_SetChnParam ret=0x%x\n", ret);
      CVI_VDEC_DetachVbPool(vdec_chn_);
      CVI_VDEC_DestroyChn(vdec_chn_);
      goto err_pools;
    }
  }

  ret = CVI_VDEC_StartRecvStream(vdec_chn_);
  if (ret != 0) {
    LOGE("[VdecRtsp] CVI_VDEC_StartRecvStream ret=0x%x\n", ret);
    CVI_VDEC_DetachVbPool(vdec_chn_);
    CVI_VDEC_DestroyChn(vdec_chn_);
    goto err_pools;
  }

  LOGI("[VdecRtsp] VDEC chn=%d  max=%ux%u  stream=%ux%u  %s  "
       "stream_buf=%u  frame_cnt=%u  VB_SOURCE_USER  pic_pool=%u  tmv_pool=%u\n",
       vdec_chn_, max_w, max_h, width, height, is_h265 ? "H265" : "H264",
       stream_buf_size, kFrameBufCnt, vb_pool_, vb_tmv_pool_);
  return true;

err_pools:
  if (vb_tmv_pool_ != VB_INVALID_POOLID) {
    CVI_VB_DestroyPool(vb_tmv_pool_);
    vb_tmv_pool_ = VB_INVALID_POOLID;
  }
  if (vb_pool_ != VB_INVALID_POOLID) {
    CVI_VB_DestroyPool(vb_pool_);
    vb_pool_ = VB_INVALID_POOLID;
  }
  return false;
}

void PyRtspClientVdec::deinitVdec() {
  if (vdec_chn_ < 0) return;
  CVI_VDEC_StopRecvStream(vdec_chn_);
  CVI_VDEC_ResetChn(vdec_chn_);    // flush VB blocks back to pool before destroy
  CVI_VDEC_DetachVbPool(vdec_chn_);
  CVI_VDEC_DestroyChn(vdec_chn_);
  if (vb_tmv_pool_ != VB_INVALID_POOLID) {
    CVI_VB_DestroyPool(vb_tmv_pool_);
    vb_tmv_pool_ = VB_INVALID_POOLID;
  }
  if (vb_pool_ != VB_INVALID_POOLID) {
    CVI_VB_DestroyPool(vb_pool_);
    vb_pool_ = VB_INVALID_POOLID;
  }
  LOGI("[VdecRtsp] VDEC chn=%d destroyed\n", vdec_chn_);
}

PyImage PyRtspClientVdec::read() {
  if (closed_)
    throw std::runtime_error("RtspClientVdec: stream is closed");

  // ── H265 fallback: OpenCV software decode ────────────────────────────────
  if (use_cv_fallback_) {
    cv::Mat bgr;
    if (!cv_cap_.read(bgr) || bgr.empty())
      throw std::runtime_error(
          "RtspClientVdec: H265 OpenCV fallback: falha ao ler frame");

    if (target_width_ > 0 && target_height_ > 0 &&
        (bgr.cols != target_width_ || bgr.rows != target_height_))
      cv::resize(bgr, bgr, cv::Size(target_width_, target_height_));

    const uint32_t w = static_cast<uint32_t>(bgr.cols);
    const uint32_t h = static_cast<uint32_t>(bgr.rows);

    cv::Mat i420;
    cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

    auto vpss = std::make_shared<VPSSImage>(w, h, ImageFormat::YUV420SP_UV,
                                            TDLDataType::UINT8, true);
    std::vector<uint8_t*> dst_ptrs = vpss->getVirtualAddress();
    std::vector<uint32_t> strides  = vpss->getStrides();

    if (dst_ptrs.size() < 2 || !dst_ptrs[0] || !dst_ptrs[1])
      throw std::runtime_error("RtspClientVdec: falha ao mapear VPSSImage");

    // Y plane
    const uint8_t* y_src = i420.data;
    for (uint32_t row = 0; row < h; ++row)
      std::memcpy(dst_ptrs[0] + row * strides[0], y_src + row * w, w);

    // UV interleaved (NV12 = UVUV…)
    const uint8_t* u_src = i420.data + (size_t)h * w;
    const uint8_t* v_src = i420.data + (size_t)h * w * 5 / 4;
    const uint32_t hw = w / 2, hh = h / 2;
    for (uint32_t row = 0; row < hh; ++row) {
      uint8_t*       d = dst_ptrs[1] + row * strides[1];
      const uint8_t* u = u_src + row * hw;
      const uint8_t* v = v_src + row * hw;
      for (uint32_t col = 0; col < hw; ++col) {
        d[col * 2]     = u[col];
        d[col * 2 + 1] = v[col];
      }
    }
    vpss->flushCache();
    std::shared_ptr<BaseImage> base = vpss;
    return PyImage(base);
  }

  // ── H264: VDEC hardware decode ────────────────────────────────────────────
  if (frame_held_)
    throw std::runtime_error(
        "RtspClientVdec: call release() before the next read()");

  // CVI_VDEC_GetFrame is NOT a blocking wait: it checks whether a decoded
  // frame is already in the display queue and returns CVI_ERR_VDEC_ERR_INVALID_RET
  // (0xc0058041) immediately if the queue is empty.  We must poll until a frame
  // arrives or the caller-specified timeout elapses.
  constexpr int kPollIntervalMs = 2;
  auto deadline = std::chrono::steady_clock::now()
                + std::chrono::milliseconds(timeout_ms_);

  VIDEO_FRAME_INFO_S frame{};
  int ret = 0;
  while (true) {
    ret = CVI_VDEC_GetFrame(vdec_chn_, &frame, 0);
    if (ret == 0) break;  // got a frame

    // 0xc0058041 = CVI_ERR_VDEC_ERR_INVALID_RET → display queue empty, retry
    if (ret == static_cast<int>(0xc0058041u) &&
        std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    // Real error or deadline exceeded
    char msg[128];
    snprintf(msg, sizeof(msg),
             "RtspClientVdec: CVI_VDEC_GetFrame failed ret=0x%x (chn=%d timeout=%dms)",
             ret, vdec_chn_, timeout_ms_);
    throw std::runtime_error(msg);
  }

  // Map virtual addresses if the VDEC driver left them as NULL.
  for (int i = 0; i < 3; ++i) {
    if (frame.stVFrame.u32Length[i] != 0 &&
        frame.stVFrame.pu8VirAddr[i] == nullptr) {
      frame.stVFrame.pu8VirAddr[i] = static_cast<CVI_U8*>(
          CVI_SYS_Mmap(frame.stVFrame.u64PhyAddr[i],
                       frame.stVFrame.u32Length[i]));
      CVI_SYS_IonFlushCache(frame.stVFrame.u64PhyAddr[i],
                             frame.stVFrame.pu8VirAddr[i],
                             frame.stVFrame.u32Length[i]);
    }
  }

  held_frame_ = frame;
  frame_held_ = true;

  std::shared_ptr<BaseImage> base = std::make_shared<VPSSImage>(frame);
  return PyImage(base);
}

void PyRtspClientVdec::release() {
  if (use_cv_fallback_) return;  // OpenCV path: no VB frame to release

  if (!frame_held_) return;

  for (int i = 0; i < 3; ++i) {
    if (held_frame_.stVFrame.u32Length[i] != 0 &&
        held_frame_.stVFrame.pu8VirAddr[i] != nullptr) {
      CVI_SYS_Munmap(static_cast<void*>(held_frame_.stVFrame.pu8VirAddr[i]),
                     held_frame_.stVFrame.u32Length[i]);
      held_frame_.stVFrame.pu8VirAddr[i] = nullptr;
    }
  }

  int ret = CVI_VDEC_ReleaseFrame(vdec_chn_, &held_frame_);
  if (ret != 0)
    LOGE("[VdecRtsp] CVI_VDEC_ReleaseFrame ret=0x%x\n", ret);

  frame_held_ = false;
}

void PyRtspClientVdec::pinForInference() {
  if (use_cv_fallback_) return;
  if (!frame_held_) {
    LOGW("[VdecRtsp] pinForInference: no current frame held — ignoring\n");
    return;
  }
  if (infer_frame_held_) {
    LOGW("[VdecRtsp] pinForInference: previous inference frame not yet released — ignoring\n");
    return;
  }
  infer_frame_       = held_frame_;
  infer_frame_held_  = true;
  held_frame_        = {};
  frame_held_        = false;
}

void PyRtspClientVdec::releaseInference() {
  if (use_cv_fallback_) return;
  if (!infer_frame_held_) return;

  for (int i = 0; i < 3; ++i) {
    if (infer_frame_.stVFrame.u32Length[i] != 0 &&
        infer_frame_.stVFrame.pu8VirAddr[i] != nullptr) {
      CVI_SYS_Munmap(static_cast<void*>(infer_frame_.stVFrame.pu8VirAddr[i]),
                     infer_frame_.stVFrame.u32Length[i]);
      infer_frame_.stVFrame.pu8VirAddr[i] = nullptr;
    }
  }
  int ret = CVI_VDEC_ReleaseFrame(vdec_chn_, &infer_frame_);
  if (ret != 0)
    LOGE("[VdecRtsp] releaseInference CVI_VDEC_ReleaseFrame ret=0x%x\n", ret);
  infer_frame_held_ = false;
}

void PyRtspClientVdec::close() {
  if (closed_) return;
  closed_ = true;

  event_watch_ = 1;
  if (event_thread_.joinable()) event_thread_.join();

  if (use_cv_fallback_) {
    cv_cap_.release();
  } else {
    releaseInference();
    release();
    deinitVdec();
  }
  is_opened_ = false;
  LOGI("[VdecRtsp] closed\n");
}

bool PyRtspClientVdec::isOpened() const {
  return is_opened_ && !closed_;
}

}  // namespace pytdl
