#!/bin/sh
# Milkv Duo S real poweroff.
#
# `poweroff`/`halt -p` sozinhos reiniciam o chip: a FSM do RTC corta VDD_CORE,
# mas wake sources sticky em RTC_EN_PWR_WAKEUP (bits setados por rtc_set_alarm
# que nunca são limpos) + VBAT_DET disparam cold-boot imediato porque o USB-C
# mantém RTC_VDDIO sempre alto.
#
# Zerar as duas máscaras antes do shutdown resolve. O chip só religa ao
# desconectar/reconectar o USB-C.
#
# Descoberto em 2026-04-23 via decodificação de st_on_reason/st_off_reason
# do FSBL (COLD_BOOT com wake source bit 2=VBAT_DET disparando).

devmem 0x050260BC 32 0x00000000   # RTC_EN_PWR_WAKEUP — zera todas wake sources
devmem 0x050260D0 32 0x00000000   # RTC_EN_PWR_VBAT_DET — desarma detecção de VBAT

exec /sbin/poweroff "$@"
