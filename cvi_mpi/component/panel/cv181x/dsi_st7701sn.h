#ifndef _MIPI_TX_PARAM_ST7701SN_H_
#define _MIPI_TX_PARAM_ST7701SN_H_

#include <linux/cvi_comm_mipi_tx.h>

#define ST7701SN_HACT		480
#define ST7701SN_HSA		8
#define ST7701SN_HBP		160
#define ST7701SN_HFP		140

#define ST7701SN_VACT		480
#define ST7701SN_VSA		20
#define ST7701SN_VBP		20
#define ST7701SN_VFP		8

#define PIXEL_CLK(x) ((x##_VACT + x##_VSA + x##_VBP + x##_VFP) \
	* (x##_HACT + x##_HSA + x##_HBP + x##_HFP) * 60 / 1000)

struct combo_dev_cfg_s dev_cfg_st7701sn_480x480 = {
	.devno = 0,
	.lane_id = {MIPI_TX_LANE_0, MIPI_TX_LANE_1, MIPI_TX_LANE_CLK, -1, -1},
	.lane_pn_swap = {false, false, false, false, false},
	.output_mode = OUTPUT_MODE_DSI_VIDEO,
	.video_mode = BURST_MODE,
	.output_format = OUT_FORMAT_RGB_24_BIT,
	.sync_info = {
		.vid_hsa_pixels = ST7701SN_HSA,
		.vid_hbp_pixels = ST7701SN_HBP,
		.vid_hfp_pixels = ST7701SN_HFP,
		.vid_hline_pixels = ST7701SN_HACT,
		.vid_vsa_lines = ST7701SN_VSA,
		.vid_vbp_lines = ST7701SN_VBP,
		.vid_vfp_lines = ST7701SN_VFP,
		.vid_active_lines = ST7701SN_VACT,
		.vid_vsa_pos_polarity = false,
		.vid_hsa_pos_polarity = false,
	},
	.pixel_clk = PIXEL_CLK(ST7701SN),
};

const struct hs_settle_s hs_timing_cfg_st7701sn_480x480 = { .prepare = 6, .zero = 32, .trail = 1 };

static CVI_U8 data_st7701sn_0[]  = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x13 };
static CVI_U8 data_st7701sn_1[]  = { 0xEF, 0x08 };
static CVI_U8 data_st7701sn_2[]  = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x10 };
static CVI_U8 data_st7701sn_3[]  = { 0xC0, 0x3B, 0x00 };
static CVI_U8 data_st7701sn_4[]  = { 0xC1, 0x0B, 0x02 };
static CVI_U8 data_st7701sn_5[]  = { 0xC2, 0x00, 0x02 };
static CVI_U8 data_st7701sn_6[]  = { 0xCC, 0x10 };
static CVI_U8 data_st7701sn_7[]  = { 0xB0, 0x40, 0x06, 0x4F, 0x01, 0x04, 0x00, 0x00, 0x00, 0x06, 0x1D, 0x00, 0x0C, 0x0C, 0x69, 0x36, 0x24 };
static CVI_U8 data_st7701sn_8[]  = { 0xB1, 0x40, 0x1B, 0x64, 0x16, 0x19, 0x0A, 0x07, 0x0C, 0x0B, 0x30, 0x0A, 0x1B, 0x17, 0x7E, 0x3B, 0x1A };
static CVI_U8 data_st7701sn_9[]  = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x11 };
static CVI_U8 data_st7701sn_10[] = { 0xB0, 0x6D };
static CVI_U8 data_st7701sn_11[] = { 0xB1, 0x0D };
static CVI_U8 data_st7701sn_12[] = { 0xB2, 0x81 };
static CVI_U8 data_st7701sn_13[] = { 0xB3, 0x80 };
static CVI_U8 data_st7701sn_14[] = { 0xB5, 0x43 };
static CVI_U8 data_st7701sn_15[] = { 0xB7, 0x85 };
static CVI_U8 data_st7701sn_16[] = { 0xB8, 0x20 };
static CVI_U8 data_st7701sn_17[] = { 0xC1, 0x78 };
static CVI_U8 data_st7701sn_18[] = { 0xC2, 0x78 };
static CVI_U8 data_st7701sn_19[] = { 0xD0, 0x88 };
static CVI_U8 data_st7701sn_20[] = { 0xE0, 0x00, 0x00, 0x02 };
static CVI_U8 data_st7701sn_21[] = { 0xE1, 0x03, 0xA0, 0x00, 0x00, 0x04, 0xA0, 0x00, 0x00, 0x00, 0x20, 0x20 };
static CVI_U8 data_st7701sn_22[] = { 0xE2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static CVI_U8 data_st7701sn_23[] = { 0xE3, 0x00, 0x00, 0x11, 0x00 };
static CVI_U8 data_st7701sn_24[] = { 0xE4, 0x22, 0x00 };
static CVI_U8 data_st7701sn_25[] = { 0xE5, 0x05, 0xEC, 0xA0, 0xA0, 0x07, 0xEE, 0xA0, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static CVI_U8 data_st7701sn_26[] = { 0xE6, 0x00, 0x00, 0x11, 0x00 };
static CVI_U8 data_st7701sn_27[] = { 0xE7, 0x22, 0x00 };
static CVI_U8 data_st7701sn_28[] = { 0xE8, 0x06, 0xED, 0xA0, 0xA0, 0x08, 0xEF, 0xA0, 0xA0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static CVI_U8 data_st7701sn_29[] = { 0xEB, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x00 };
static CVI_U8 data_st7701sn_30[] = { 0xED, 0xFF, 0xFF, 0xFF, 0xBA, 0x0A, 0xBF, 0x45, 0xFF, 0xFF, 0x54, 0xFB, 0xA0, 0xAB, 0xFF, 0xFF, 0xFF };
static CVI_U8 data_st7701sn_31[] = { 0xEF, 0x10, 0x0D, 0x04, 0x08, 0x3F, 0x1F };
static CVI_U8 data_st7701sn_32[] = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x13 };
static CVI_U8 data_st7701sn_33[] = { 0xE8, 0x00, 0x0E };
static CVI_U8 data_st7701sn_34[] = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x00 };

static CVI_U8 data_st7701sn_35[] = { 0x11 };

static CVI_U8 data_st7701sn_36[] = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x13 };
static CVI_U8 data_st7701sn_37[] = { 0xE8, 0x00, 0x0C };
static CVI_U8 data_st7701sn_38[] = { 0xE8, 0x00, 0x00 };
static CVI_U8 data_st7701sn_39[] = { 0xFF, 0x77, 0x01, 0x00, 0x00, 0x00 };

static CVI_U8 data_st7701sn_40[] = { 0x29 }; // Display ON

const struct dsc_instr dsi_init_cmds_st7701sn_480x480[] = {
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_0 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_1 },
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_2 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_3 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_4 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_5 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_6 },
    {.delay = 0,   .data_type = 0x29, .size = 17, .data = data_st7701sn_7 },
    {.delay = 0,   .data_type = 0x29, .size = 17, .data = data_st7701sn_8 },
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_9 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_10 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_11 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_12 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_13 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_14 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_15 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_16 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_17 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_18 },
    {.delay = 0,   .data_type = 0x15, .size = 2,  .data = data_st7701sn_19 },
    {.delay = 0,   .data_type = 0x29, .size = 4,  .data = data_st7701sn_20 },
    {.delay = 0,   .data_type = 0x29, .size = 12, .data = data_st7701sn_21 },
    {.delay = 0,   .data_type = 0x29, .size = 14, .data = data_st7701sn_22 },
    {.delay = 0,   .data_type = 0x29, .size = 5,  .data = data_st7701sn_23 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_24 },
    {.delay = 0,   .data_type = 0x29, .size = 17, .data = data_st7701sn_25 },
    {.delay = 0,   .data_type = 0x29, .size = 5,  .data = data_st7701sn_26 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_27 },
    {.delay = 0,   .data_type = 0x29, .size = 17, .data = data_st7701sn_28 },
    {.delay = 0,   .data_type = 0x29, .size = 8,  .data = data_st7701sn_29 },
    {.delay = 0,   .data_type = 0x29, .size = 17, .data = data_st7701sn_30 },
    {.delay = 0,   .data_type = 0x29, .size = 7,  .data = data_st7701sn_31 },
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_32 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_33 },
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_34 },

    {.delay = 120, .data_type = 0x05, .size = 1,  .data = data_st7701sn_35 },

    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_36 },
    {.delay = 10,  .data_type = 0x29, .size = 3,  .data = data_st7701sn_37 },
    {.delay = 0,   .data_type = 0x29, .size = 3,  .data = data_st7701sn_38 },
    {.delay = 0,   .data_type = 0x29, .size = 6,  .data = data_st7701sn_39 },

    {.delay = 50,  .data_type = 0x05, .size = 1,  .data = data_st7701sn_40 }
};

#else
#error "MIPI_TX_PARAM multi-delcaration!!"
#endif // _MIPI_TX_PARAM_ST7701SN_H_
