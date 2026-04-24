#ifndef _VIDEO_RECORDER_HPP_
#define _VIDEO_RECORDER_HPP_

#include <cvi_type.h>
#include <cvi_venc.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct AVFormatContext;
struct AVStream;

// VideoRecorder
// --------------
// Encodes incoming VIDEO_FRAME_INFO_S frames with VENC (H.264 CBR) and muxes
// them to MP4 segments via libavformat.  Segment rotation is time-based:
// after `segment_seconds` of wall-clock recording we request an IDR, close
// the current MP4, and open a new one.  The new segment starts on the next
// IDR keyframe so it is independently seekable/playable.
//
// Filenames are derived from the recording start time of each segment:
//   <out_dir>/dvr_YYYY-MM-DD_HH-MM-SS.mp4
//
// Files produced per segment:
//   dvr_<ts>.mp4   H.264 video, playable in any HTML5 <video>
//   dvr_<ts>.jpg   thumbnail captured from the first frame of the segment
//                  (JPEG encoded by VENC on a separate channel)
//
// Channel usage: the recorder consumes one VENC channel (default chn=0).
// Thumbnails use jpeg_chn (default 1).  Do NOT reuse these channels for an
// RTSP server at the same time — pass chn=2/3 if you need both.
class VideoRecorder {
 public:
  struct SegmentInfo {
    std::string filename;      // basename only, e.g. "dvr_2026-04-20_14-30-00.mp4"
    std::string thumbnail;     // basename only, e.g. "dvr_2026-04-20_14-30-00.jpg"
    int64_t     size_bytes  = 0;
    int64_t     started_ms  = 0;   // epoch ms when segment started
    int64_t     duration_ms = 0;   // wall-clock duration
    int         frame_count = 0;
  };

  VideoRecorder(int32_t width, int32_t height,
                const std::string& out_dir,
                const std::string& codec = "h264",
                int32_t segment_seconds = 30,
                int32_t fps = 15,
                int32_t bitrate_kbps = 3072,
                int32_t gop = 15,
                int32_t chn = 0,
                int32_t jpeg_chn = 1);
  ~VideoRecorder();

  // Encode + mux one frame.  Returns 0 on success, non-zero on error.
  // The frame is NOT consumed — the caller retains ownership and must
  // call cam.release() after sendFrame returns.
  int32_t sendFrame(VIDEO_FRAME_INFO_S* frame);

  // Force closing the current segment now.  The next frame opens a new one.
  void rotate();

  // Current segment basename, or "" if no segment is open.
  std::string currentSegment() const;

  // Wall-clock epoch time (ms) when the current segment was opened.
  // Returns 0 while no segment is open. Use to align sidecar files
  // (detections, captions) with the MP4 PTS timeline.
  int64_t segmentStartMs() const;

  // Output directory (absolute path as provided to the constructor).
  std::string outputDir() const { return out_dir_; }

  // Closed-segments history since the recorder was created.
  std::vector<SegmentInfo> history() const;

 private:
  // VENC lifecycle
  int32_t initVENC();
  int32_t destroyVENC();

  // Segment file lifecycle
  int32_t openSegment();
  int32_t closeSegment();

  // Pull encoded packets from VENC and write to the current MP4.
  int32_t drainEncoder(VIDEO_FRAME_INFO_S* source_for_thumb);

  // Extract SPS+PPS (H.264) or VPS+SPS+PPS (H.265) from Annex B data.
  // Result is stored into extradata_ as a concatenated AnnexB blob
  // (start codes preserved) — libavformat's MP4 muxer auto-converts it
  // to AVCC/HVCC.
  bool collectExtradata(const uint8_t* annex_b, int len);

  // Write one JPEG thumbnail for the current segment (only first call
  // per segment actually writes — subsequent calls are no-ops).
  void writeThumbnailOnce(VIDEO_FRAME_INFO_S* frame);

  // Timestamped filename base (no extension), e.g. "dvr_2026-04-20_14-30-00".
  static std::string makeBasename(int64_t epoch_ms);

  // ─────────────────────────────────────────────────────────────────────
  int32_t        chn_;
  int32_t        jpeg_chn_;
  PAYLOAD_TYPE_E payload_;
  int32_t        width_;
  int32_t        height_;
  int32_t        fps_;
  int32_t        gop_;
  int32_t        bitrate_;
  int32_t        segment_seconds_;
  std::string    out_dir_;

  // Current segment state
  AVFormatContext* fmt_ctx_    = nullptr;
  AVStream*        av_stream_  = nullptr;
  int64_t          seg_start_ms_  = 0;
  int64_t          seg_frame_idx_ = 0;
  int64_t          seg_last_pts_  = -1;  // last PTS written (in stream time base)
  int              seg_frame_count_ = 0;
  std::string      seg_basename_;
  bool             seg_thumb_done_ = false;
  bool             rotate_pending_ = false;  // IDR requested, waiting for keyframe
  bool             venc_ready_ = false;

  std::vector<uint8_t> extradata_;   // concatenated SPS+PPS (Annex B)

  // Closed segments
  mutable std::mutex        history_mutex_;
  std::vector<SegmentInfo>  history_;
};

#endif  // _VIDEO_RECORDER_HPP_
