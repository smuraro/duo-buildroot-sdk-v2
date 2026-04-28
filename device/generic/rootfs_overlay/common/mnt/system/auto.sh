#!/bin/sh
# Executado no boot pelo S99user (em background).
# Inicia o DVR (sample_dvr.py) assim que o SD card estiver montado.

# O S99user só exporta USERDATAPATH/SYSTEMPATH — as libs do TPU/tdl_sdk
# ficam em /mnt/system/lib, /mnt/system/usr/lib etc, mas sem isso no
# LD_LIBRARY_PATH o `import tdl` falha com "libtdl_core.so not found".
# Espelha o que o login interativo já configura (ver /etc/profile).
export LD_LIBRARY_PATH="/lib:/lib/3rd:/usr/lib:/usr/local/lib:/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:/mnt/data/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/mnt/system/usr/bin:/mnt/system/usr/sbin:/mnt/data/bin:/mnt/data/sbin${PATH:+:$PATH}"

DVR_SCRIPT=/mnt/system/usr/bin/python/sample_dvr.py
MOUNTPOINT=/mnt/sd
LOG_PRIMARY=/mnt/sd/dvr/dvr.log
LOG_FALLBACK=/tmp/dvr.log

# Flag persistente em /mnt/data (userdata, sobrevive reboot).
# Default: ausente -> não inicia. Use dvr-enable / dvr-disable para alternar.
ENABLE_FLAG=/mnt/data/dvr_enabled
[ -f "$ENABLE_FLAG" ] || exit 0

[ -f "$DVR_SCRIPT" ] || exit 0

# Espera até 30s pelo SD card. O próprio script também espera 10s, mas
# falhamos cedo com log útil se nunca montar.
i=0
while ! mountpoint -q "$MOUNTPOINT" 2>/dev/null && [ $i -lt 30 ]; do
  sleep 1
  i=$((i+1))
done

if mountpoint -q "$MOUNTPOINT" 2>/dev/null; then
  mkdir -p /mnt/sd/dvr 2>/dev/null
  LOG="$LOG_PRIMARY"

  # Clock rollover: sem RTC, o relógio zera a cada boot e arquivos novos
  # colidem com gravações anteriores (mesmo nome). Avança o clock para
  # 1 minuto após o arquivo mais recente no SD card, mantendo nomes
  # monotonicamente crescentes entre reboots. No-op quando o clock
  # corrente já está à frente (ex: futuramente com RTC ou NTP).
  newest=$(ls -t /mnt/sd/dvr/*.mp4 /mnt/sd/dvr/*.jpg /mnt/sd/dvr/*.jsonl \
               2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    ts=$(stat -c %Y "$newest" 2>/dev/null)
    now=$(date +%s)
    if [ -n "$ts" ] && [ "$ts" -gt "$now" ]; then
      date -s "@$((ts + 60))" >/dev/null 2>&1
      echo "[auto.sh] clock bumped from $now to $(date +%s) (ref $newest)" \
           >> "$LOG"
    fi
  fi
else
  LOG="$LOG_FALLBACK"
  echo "[auto.sh] $MOUNTPOINT não montado após 30s — logs vão para $LOG" \
       > "$LOG"
fi

echo "[auto.sh] $(date) iniciando $DVR_SCRIPT" >> "$LOG"
exec python3 "$DVR_SCRIPT" --model LSTR_DET_LANE >> "$LOG" 2>&1
