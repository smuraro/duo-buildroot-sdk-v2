#include "encoder/video_recorder/video_recorder.hpp"
#include "utils/tdl_log.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

// ─── Defaults ─────────────────────────────────────────────────────────────────
#define FRAME_MILLISEC  2000
#define MaxPicWidth     2560
#define MaxPicHeight    1440
#define BufSize         (4 * 1024 * 1024)
#define Profile         0
#define IPQpDelta       2
#define StatTime        1
// libavformat MP4 time base: 90kHz is the de-facto standard for H.264 in MP4.
static constexpr int kMp4TimeBaseHz = 90000;

// ─── Helpers ──────────────────────────────────────────────────────────────────

static int64_t nowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(
             system_clock::now().time_since_epoch()).count();
}

static bool makeDirs(const std::string& path) {
  if (path.empty()) return false;
  struct stat st;
  if (stat(path.c_str(), &st) == 0) return (st.st_mode & S_IFDIR) != 0;
  // Create parents first.
  std::string p;
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '/' && i > 0) {
      p = path.substr(0, i);
      if (stat(p.c_str(), &st) != 0) mkdir(p.c_str(), 0755);
    }
  }
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// Find the next NAL unit in an Annex B stream.  Returns pointer to the byte
// right after the start code, and out_size = bytes until the next start code
// (or end of buffer).  Returns nullptr when no more NAL units.
static const uint8_t* nextNalU(const uint8_t* data, int len,
                               int* nal_size, int* sc_len) {
  for (int i = 0; i + 3 < len; ++i) {
    if (data[i] == 0 && data[i+1] == 0) {
      int sc = 0;
      if (data[i+2] == 1) sc = 3;
      else if (data[i+2] == 0 && data[i+3] == 1) sc = 4;
      if (sc == 0) continue;
      // found start code at i, NAL begins at i+sc
      int nal_begin = i + sc;
      // find next start code
      int j = nal_begin;
      for (; j + 3 < len; ++j) {
        if (data[j] == 0 && data[j+1] == 0 &&
            (data[j+2] == 1 || (data[j+2] == 0 && data[j+3] == 1)))
          break;
      }
      int end = (j + 3 < len) ? j : len;
      *nal_size = end - nal_begin;
      *sc_len   = sc;
      return data + nal_begin;
    }
  }
  return nullptr;
}

// ─── VideoRecorder ────────────────────────────────────────────────────────────

std::string VideoRecorder::makeBasename(int64_t epoch_ms) {
  time_t  t  = (time_t)(epoch_ms / 1000);
  struct tm tm;
  localtime_r(&t, &tm);
  char buf[64];
  strftime(buf, sizeof(buf), "dvr_%Y-%m-%d_%H-%M-%S", &tm);
  return std::string(buf);
}

VideoRecorder::VideoRecorder(int32_t width, int32_t height,
                             const std::string& out_dir,
                             const std::string& codec,
                             int32_t segment_seconds,
                             int32_t fps, int32_t bitrate_kbps,
                             int32_t gop,
                             int32_t chn, int32_t jpeg_chn)
    : chn_(chn), jpeg_chn_(jpeg_chn),
      width_(width), height_(height),
      fps_(fps > 0 ? fps : 15),
      gop_(gop > 0 ? gop : 15),
      bitrate_(bitrate_kbps > 0 ? bitrate_kbps : 3072),
      segment_seconds_(segment_seconds > 0 ? segment_seconds : 30),
      out_dir_(out_dir) {
  if (codec == "h265" || codec == "H265") payload_ = PT_H265;
  else if (codec == "h264" || codec == "H264") payload_ = PT_H264;
  else {
    LOGE("VideoRecorder: invalid codec '%s' (must be h264 or h265)",
         codec.c_str());
    payload_ = PT_H264;
  }

  if (!makeDirs(out_dir_)) {
    LOGE("VideoRecorder: failed to create output dir '%s'", out_dir_.c_str());
  }

  if (initVENC() != 0) {
    LOGE("VideoRecorder: failed to initialize VENC");
    return;
  }
  venc_ready_ = true;
  LOGI("VideoRecorder: chn=%d %dx%d %s %dkbps gop=%d fps=%d seg=%ds dir=%s",
       chn_, width_, height_, codec.c_str(), bitrate_, gop_, fps_,
       segment_seconds_, out_dir_.c_str());
}

VideoRecorder::~VideoRecorder() {
  if (fmt_ctx_) closeSegment();
  if (venc_ready_) destroyVENC();
}

int32_t VideoRecorder::initVENC() {
  VENC_CHN_ATTR_S attr;
  memset(&attr, 0, sizeof(attr));
  attr.stVencAttr.u32PicWidth     = width_;
  attr.stVencAttr.u32PicHeight    = height_;
  attr.stVencAttr.u32MaxPicWidth  = MaxPicWidth;
  attr.stVencAttr.u32MaxPicHeight = MaxPicHeight;
  attr.stVencAttr.u32BufSize      = BufSize;
  attr.stVencAttr.u32Profile      = Profile;
  attr.stVencAttr.enType          = payload_;
  attr.stGopAttr.enGopMode        = VENC_GOPMODE_NORMALP;
  attr.stGopAttr.stNormalP.s32IPQpDelta = IPQpDelta;

  if (payload_ == PT_H264) {
    attr.stVencAttr.stAttrH264e.bSingleLumaBuf  = CVI_FALSE;
    attr.stVencAttr.stAttrH264e.bRcnRefShareBuf = CVI_FALSE;
    attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
    attr.stRcAttr.stH264Cbr.u32Gop            = gop_;
    attr.stRcAttr.stH264Cbr.u32StatTime       = StatTime;
    attr.stRcAttr.stH264Cbr.fr32DstFrameRate  = fps_;
    attr.stRcAttr.stH264Cbr.u32SrcFrameRate   = fps_;
    attr.stRcAttr.stH264Cbr.u32BitRate        = bitrate_;
    attr.stRcAttr.stH264Cbr.bVariFpsEn        = CVI_TRUE;
  } else {
    attr.stVencAttr.stAttrH265e.bRcnRefShareBuf = CVI_FALSE;
    attr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
    attr.stRcAttr.stH265Cbr.u32Gop            = gop_;
    attr.stRcAttr.stH265Cbr.u32StatTime       = StatTime;
    attr.stRcAttr.stH265Cbr.fr32DstFrameRate  = fps_;
    attr.stRcAttr.stH265Cbr.u32SrcFrameRate   = fps_;
    attr.stRcAttr.stH265Cbr.u32BitRate        = bitrate_;
    attr.stRcAttr.stH265Cbr.bVariFpsEn        = CVI_TRUE;
  }

  int ret = CVI_VENC_CreateChn(chn_, &attr);
  if (ret != 0) { LOGE("CVI_VENC_CreateChn chn=%d ret=0x%x", chn_, ret); return ret; }

  VENC_CHN_PARAM_S param;
  memset(&param, 0, sizeof(param));
  CVI_VENC_GetChnParam(chn_, &param);
  CVI_VENC_SetChnParam(chn_, &param);

  VENC_RECV_PIC_PARAM_S recv;
  recv.s32RecvPicNum = -1;
  ret = CVI_VENC_StartRecvFrame(chn_, &recv);
  if (ret != 0) { LOGE("CVI_VENC_StartRecvFrame chn=%d ret=0x%x", chn_, ret); return ret; }
  return 0;
}

int32_t VideoRecorder::destroyVENC() {
  CVI_VENC_StopRecvFrame(chn_);
  CVI_VENC_DestroyChn(chn_);
  return 0;
}

int32_t VideoRecorder::openSegment() {
  if (fmt_ctx_) return 0;
  if (extradata_.empty()) return 0;  // wait for SPS/PPS first

  seg_start_ms_    = nowMs();
  seg_frame_idx_   = 0;
  seg_last_pts_    = -1;
  seg_frame_count_ = 0;
  seg_thumb_done_ = false;
  seg_basename_   = makeBasename(seg_start_ms_);
  std::string path = out_dir_ + "/" + seg_basename_ + ".mp4";

  const char* fmt_name = "mp4";
  int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, fmt_name, path.c_str());
  if (ret < 0 || !fmt_ctx_) {
    LOGE("avformat_alloc_output_context2 failed: %d", ret);
    return -1;
  }

  av_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
  if (!av_stream_) { LOGE("avformat_new_stream failed"); return -1; }

  av_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  av_stream_->codecpar->codec_id   =
      (payload_ == PT_H265) ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
  av_stream_->codecpar->width      = width_;
  av_stream_->codecpar->height     = height_;
  av_stream_->codecpar->format     = AV_PIX_FMT_YUV420P;
  av_stream_->codecpar->codec_tag  = 0;  // let muxer pick (avc1/hev1)
  av_stream_->time_base            = (AVRational){1, kMp4TimeBaseHz};
  av_stream_->avg_frame_rate       = (AVRational){fps_, 1};
  av_stream_->r_frame_rate         = (AVRational){fps_, 1};

  // Extradata in Annex B form — libavformat MP4 muxer converts it to
  // AVCC (H.264) / HVCC (H.265) when writing the avcC/hvcC box.
  av_stream_->codecpar->extradata =
      (uint8_t*)av_mallocz(extradata_.size() + AV_INPUT_BUFFER_PADDING_SIZE);
  if (!av_stream_->codecpar->extradata) return -1;
  memcpy(av_stream_->codecpar->extradata, extradata_.data(), extradata_.size());
  av_stream_->codecpar->extradata_size = (int)extradata_.size();

  ret = avio_open(&fmt_ctx_->pb, path.c_str(), AVIO_FLAG_WRITE);
  if (ret < 0) {
    LOGE("avio_open %s failed: %d", path.c_str(), ret);
    avformat_free_context(fmt_ctx_);
    fmt_ctx_ = nullptr;
    return -1;
  }

  // Note: +faststart forced a two-pass close (rewriting the whole file to
  // move moov to the beginning). On SD+NTFS-3G that became a multi-second
  // hang on Ctrl+C and segment rotation. Writing moov at the end is fully
  // valid MP4 and plays fine in browsers (they read the last bytes to find
  // the moov offset, then seek to the start).
  ret = avformat_write_header(fmt_ctx_, nullptr);
  if (ret < 0) {
    LOGE("avformat_write_header failed: %d", ret);
    avio_closep(&fmt_ctx_->pb);
    avformat_free_context(fmt_ctx_);
    fmt_ctx_ = nullptr;
    return -1;
  }

  LOGI("VideoRecorder: opened segment %s", path.c_str());
  return 0;
}

int32_t VideoRecorder::closeSegment() {
  if (!fmt_ctx_) return 0;
  av_write_trailer(fmt_ctx_);
  if (fmt_ctx_->pb) avio_closep(&fmt_ctx_->pb);

  SegmentInfo info;
  info.filename    = seg_basename_ + ".mp4";
  info.thumbnail   = seg_thumb_done_ ? (seg_basename_ + ".jpg") : "";
  info.started_ms  = seg_start_ms_;
  info.duration_ms = nowMs() - seg_start_ms_;
  info.frame_count = seg_frame_count_;
  std::string full = out_dir_ + "/" + info.filename;
  struct stat st;
  if (stat(full.c_str(), &st) == 0) info.size_bytes = (int64_t)st.st_size;

  // fsync do MP4 recém-fechado: garante que o segmento finalizado está
  // fisicamente no disco antes do próximo ser iniciado. Em ext4 com
  // commit=5s, evita perder o último segmento em queda de energia.
  // Custo: algumas dezenas de ms por rotação; desprezível vs. 30s de
  // gravação e só acontece a cada rotação.
  int fd = ::open(full.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  }

  avformat_free_context(fmt_ctx_);
  fmt_ctx_   = nullptr;
  av_stream_ = nullptr;

  {
    std::lock_guard<std::mutex> lk(history_mutex_);
    history_.push_back(info);
  }
  LOGI("VideoRecorder: closed segment %s (%lld bytes, %d frames, %.1fs)",
       info.filename.c_str(), (long long)info.size_bytes,
       info.frame_count, info.duration_ms / 1000.0);
  return 0;
}

void VideoRecorder::rotate() { rotate_pending_ = true; }

bool VideoRecorder::collectExtradata(const uint8_t* annex_b, int len) {
  if (!extradata_.empty()) return true;
  // Collect SPS (NAL type 7 for H.264, 33 for H.265) and
  //         PPS (NAL type 8 for H.264, 34 for H.265), plus
  //         VPS (NAL type 32 for H.265).
  bool have_sps = false, have_pps = false, have_vps = (payload_ == PT_H264);

  const uint8_t* p = annex_b;
  int remaining = len;
  std::vector<uint8_t> acc;

  while (remaining > 0) {
    int nal_size = 0, sc_len = 0;
    const uint8_t* nal = nextNalU(p, remaining, &nal_size, &sc_len);
    if (!nal) break;
    int nal_off = (int)(nal - annex_b);

    uint8_t nal_type;
    if (payload_ == PT_H264) nal_type = nal[0] & 0x1F;
    else                     nal_type = (nal[0] >> 1) & 0x3F;

    bool wanted = false;
    if (payload_ == PT_H264) {
      if      (nal_type == 7) { have_sps = true; wanted = true; }
      else if (nal_type == 8) { have_pps = true; wanted = true; }
    } else {
      if      (nal_type == 32) { have_vps = true; wanted = true; }
      else if (nal_type == 33) { have_sps = true; wanted = true; }
      else if (nal_type == 34) { have_pps = true; wanted = true; }
    }

    if (wanted) {
      // append start code + NAL
      acc.insert(acc.end(), annex_b + nal_off - sc_len, annex_b + nal_off);
      acc.insert(acc.end(), nal, nal + nal_size);
    }

    int consumed = (int)(nal - p) + nal_size;
    p         += consumed;
    remaining -= consumed;
  }

  if (have_sps && have_pps && have_vps) {
    extradata_ = std::move(acc);
    LOGI("VideoRecorder: collected extradata (%zu bytes)", extradata_.size());
    return true;
  }
  return false;
}

void VideoRecorder::writeThumbnailOnce(VIDEO_FRAME_INFO_S* frame) {
  if (seg_thumb_done_ || !frame) return;

  // Encode JPEG on a separate VENC channel (JPEG codec).  We create and tear
  // down the channel per-thumbnail to keep things simple — segments rotate
  // every 30s so the overhead is negligible.
  VENC_CHN_ATTR_S attr;
  memset(&attr, 0, sizeof(attr));
  attr.stVencAttr.enType           = PT_JPEG;
  attr.stVencAttr.u32MaxPicWidth   = MaxPicWidth;
  attr.stVencAttr.u32MaxPicHeight  = MaxPicHeight;
  attr.stVencAttr.u32PicWidth      = width_;
  attr.stVencAttr.u32PicHeight     = height_;
  attr.stVencAttr.u32BufSize       = 2 * 1024 * 1024;
  attr.stVencAttr.u32Profile       = 0;
  attr.stVencAttr.bByFrame         = CVI_TRUE;
  attr.stVencAttr.stAttrJpege.bSupportDCF    = CVI_FALSE;
  attr.stVencAttr.stAttrJpege.enReceiveMode  = VENC_PIC_RECEIVE_SINGLE;
  attr.stVencAttr.stAttrJpege.stMPFCfg.u8LargeThumbNailNum = 0;
  attr.stGopAttr.enGopMode                    = VENC_GOPMODE_NORMALP;

  if (CVI_VENC_CreateChn(jpeg_chn_, &attr) != 0) {
    LOGE("VideoRecorder: thumbnail CreateChn(%d) failed", jpeg_chn_);
    return;
  }
  VENC_JPEG_PARAM_S jp;
  CVI_VENC_GetJpegParam(jpeg_chn_, &jp);
  jp.u32Qfactor = 70;
  CVI_VENC_SetJpegParam(jpeg_chn_, &jp);

  VENC_RECV_PIC_PARAM_S recv;
  recv.s32RecvPicNum = 1;
  CVI_VENC_StartRecvFrame(jpeg_chn_, &recv);

  int ret = CVI_VENC_SendFrame(jpeg_chn_, frame, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("VideoRecorder: thumbnail SendFrame failed: 0x%x", ret);
    CVI_VENC_StopRecvFrame(jpeg_chn_);
    CVI_VENC_DestroyChn(jpeg_chn_);
    return;
  }

  VENC_STREAM_S stream;
  memset(&stream, 0, sizeof(stream));
  VENC_PACK_S packs[4];
  memset(packs, 0, sizeof(packs));
  stream.pstPack = packs;
  stream.u32PackCount = 4;

  ret = CVI_VENC_GetStream(jpeg_chn_, &stream, FRAME_MILLISEC);
  if (ret == 0 && stream.u32PackCount > 0) {
    std::string path = out_dir_ + "/" + seg_basename_ + ".jpg";
    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
      for (unsigned i = 0; i < stream.u32PackCount; ++i) {
        VENC_PACK_S* pk = &stream.pstPack[i];
        f.write((const char*)(pk->pu8Addr + pk->u32Offset),
                pk->u32Len - pk->u32Offset);
      }
      seg_thumb_done_ = true;
    } else {
      LOGE("VideoRecorder: failed to write thumbnail %s", path.c_str());
    }
    CVI_VENC_ReleaseStream(jpeg_chn_, &stream);
  } else {
    LOGE("VideoRecorder: thumbnail GetStream failed: 0x%x", ret);
  }

  CVI_VENC_StopRecvFrame(jpeg_chn_);
  CVI_VENC_DestroyChn(jpeg_chn_);
}

int32_t VideoRecorder::sendFrame(VIDEO_FRAME_INFO_S* frame) {
  if (!venc_ready_ || !frame) return -1;

  int ret = CVI_VENC_SendFrame(chn_, frame, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("VideoRecorder: CVI_VENC_SendFrame failed: 0x%x", ret);
    return ret;
  }

  return drainEncoder(frame);
}

int32_t VideoRecorder::drainEncoder(VIDEO_FRAME_INFO_S* source_for_thumb) {
  static constexpr uint32_t kMaxPacks = 8;
  VENC_STREAM_S stream;
  memset(&stream, 0, sizeof(stream));
  VENC_PACK_S* packs = (VENC_PACK_S*)calloc(kMaxPacks, sizeof(VENC_PACK_S));
  if (!packs) return -1;
  stream.pstPack = packs;
  stream.u32PackCount = kMaxPacks;

  int ret = CVI_VENC_GetStream(chn_, &stream, FRAME_MILLISEC);
  if (ret != 0) { free(packs); return ret; }
  if (stream.u32PackCount == 0) {
    CVI_VENC_ReleaseStream(chn_, &stream);
    free(packs);
    return 0;
  }

  // Concatenate all packs into one Annex B buffer for this frame.
  size_t total = 0;
  for (unsigned i = 0; i < stream.u32PackCount; ++i) {
    VENC_PACK_S* p = &stream.pstPack[i];
    total += (size_t)(p->u32Len - p->u32Offset);
  }
  std::vector<uint8_t> buf(total);
  uint8_t* w = buf.data();
  bool is_keyframe = false;
  for (unsigned i = 0; i < stream.u32PackCount; ++i) {
    VENC_PACK_S* p = &stream.pstPack[i];
    size_t n = p->u32Len - p->u32Offset;
    memcpy(w, p->pu8Addr + p->u32Offset, n);
    // u32DataType.enH264EType == H264E_NALU_ISLICE/IDRSLICE indicate keyframe
    // but the safest is to scan NAL types on the concatenated buffer below.
    w += n;
  }
  CVI_VENC_ReleaseStream(chn_, &stream);
  free(packs);

  // Detect keyframe by scanning NAL types (H.264: type 5 = IDR,
  // H.265: 16..21 = IRAP).
  {
    const uint8_t* p = buf.data();
    int rem = (int)buf.size();
    while (rem > 0) {
      int nal_size = 0, sc_len = 0;
      const uint8_t* nal = nextNalU(p, rem, &nal_size, &sc_len);
      if (!nal) break;
      uint8_t t = (payload_ == PT_H264) ? (nal[0] & 0x1F)
                                        : ((nal[0] >> 1) & 0x3F);
      if (payload_ == PT_H264) {
        if (t == 5) { is_keyframe = true; break; }
      } else {
        if (t >= 16 && t <= 21) { is_keyframe = true; break; }
      }
      int consumed = (int)(nal - p) + nal_size;
      p   += consumed;
      rem -= consumed;
    }
  }

  // Collect extradata from the first keyframe.
  if (extradata_.empty() && is_keyframe) {
    collectExtradata(buf.data(), (int)buf.size());
  }

  // Time-based rotation: if elapsed >= segment_seconds and not yet rotating,
  // request an IDR and mark rotate_pending_.  We close the segment at the
  // next keyframe that comes out of the encoder.
  if (fmt_ctx_ && !rotate_pending_) {
    int64_t elapsed_ms = nowMs() - seg_start_ms_;
    if (elapsed_ms >= (int64_t)segment_seconds_ * 1000) {
      CVI_VENC_RequestIDR(chn_, CVI_TRUE);
      rotate_pending_ = true;
    }
  }

  // Handle rotation boundary: on incoming keyframe, close then reopen.
  if (rotate_pending_ && is_keyframe && fmt_ctx_) {
    closeSegment();
    rotate_pending_ = false;
  }

  // Open segment lazily — we need extradata AND an incoming keyframe.
  if (!fmt_ctx_ && is_keyframe && !extradata_.empty()) {
    if (openSegment() != 0) return -1;
  }

  if (!fmt_ctx_) {
    // Still no segment (pre-roll before first keyframe) — drop.
    return 0;
  }

  // Thumbnail on first frame of segment.
  if (!seg_thumb_done_ && is_keyframe && source_for_thumb) {
    writeThumbnailOnce(source_for_thumb);
  }

  // Write AVPacket containing the frame's NAL units (Annex B).
  AVPacket* pkt = av_packet_alloc();
  if (!pkt) return -1;
  pkt->data = buf.data();
  pkt->size = (int)buf.size();
  // PTS from wall-clock elapsed since segment start — matches real time even
  // when the capture thread misses its configured fps (e.g. CPU contention).
  // Uses the segment's MP4 time base (kMp4TimeBaseHz).  Enforces strict
  // monotonicity: bumps by 1 tick if rounding produces a non-increasing value.
  int64_t elapsed_ms = nowMs() - seg_start_ms_;
  if (elapsed_ms < 0) elapsed_ms = 0;
  int64_t pts = av_rescale_q(elapsed_ms, (AVRational){1, 1000},
                             av_stream_->time_base);
  if (pts <= seg_last_pts_) pts = seg_last_pts_ + 1;
  seg_last_pts_ = pts;
  pkt->pts  = pts;
  pkt->dts  = pts;
  pkt->duration = av_rescale_q(1, (AVRational){1, fps_}, av_stream_->time_base);
  pkt->flags = is_keyframe ? AV_PKT_FLAG_KEY : 0;
  pkt->stream_index = 0;

  int wret = av_write_frame(fmt_ctx_, pkt);
  // Don't free pkt->data: we own the std::vector. Clear pkt data pointer first.
  pkt->data = nullptr;
  pkt->size = 0;
  av_packet_free(&pkt);

  if (wret < 0) {
    LOGE("VideoRecorder: av_write_frame failed: %d", wret);
    return wret;
  }

  seg_frame_idx_++;
  seg_frame_count_++;
  return 0;
}

std::string VideoRecorder::currentSegment() const {
  return fmt_ctx_ ? (seg_basename_ + ".mp4") : std::string();
}

int64_t VideoRecorder::segmentStartMs() const {
  return fmt_ctx_ ? seg_start_ms_ : 0;
}

std::vector<VideoRecorder::SegmentInfo> VideoRecorder::history() const {
  std::lock_guard<std::mutex> lk(history_mutex_);
  return history_;
}
