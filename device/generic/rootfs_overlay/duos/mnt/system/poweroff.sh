#!/bin/sh
# Milkv Duo S poweroff/suspend wrapper.
#
# Uso:
#   poweroff.sh                       # req_shdn classico (off ate reconectar USB-C)
#   poweroff.sh --alarm SEC           # suspend, acorda em SEC segundos
#
# Background:
#   No silicio CV181x/SG2000 do Duo S, ST_OFF (req_shdn) so acorda via PWR_ON
#   externo (= unplug/replug USB-C). Wake por alarm/GPIO so funciona em
#   ST_SUSPEND ("MCU-Only Mode" do datasheet sec. 5.1.2).
#
#   Caminho oficial pro --alarm: kernel CONFIG_SUSPEND -> SBI_EXT_SUSP -> opensbi
#   cvitek_sbi_system_suspend faz rtc_latch_pinmux_settings + pm_default_cv181x.
#   Dispara via "echo mem > /sys/power/state". Volta pra userspace apos wake.
#
#   Fallback (kernel sem CONFIG_SUSPEND): req_suspend manual via devmem.
#   Mata processos, monta RO, dispara FSM. Wake faz cold-boot.
#
#   Wake-on-pin via PWR_WAKEUP1 NAO funciona neste silicio — investigado
#   exaustivamente (sleep e off, todos os bits do EN_PWR_WAKEUP, REQ_ENABLES,
#   wkup_ctrl/rtc_wkup_ctrl, polaridade rejeitada pelo silicio). PWR_WAKEUP0
#   e PWR_BUTTON1 funcionam mas estao em mode 6 (LEDs do PHY).
#   Detalhes em memory/project_milkv_duos_poweroff.md.

EN_PWR_WAKEUP=0x050260BC
EN_PWR_VBAT_DET=0x050260D0
ALARM_ENABLE=0x0502600C
RTC_KO=/mnt/system/ko/cv181x_rtc.ko

# Bits do campo SLEEP wake (6:0).
ALARM_BITS=0x30        # bit 4 (PWR_BUTTON1) + bit 5 (Alarm)

# Enderecos do caminho manual (fallback).
RTC_PRDATA_SEL=0x0502603C
RTC_CTRL0_UNLOCK=0x05025004
RTC_EN_SUSPEND_REQ=0x050260E4
RTC_CTRL0=0x05025008
RTC_CTRL0_REQ_SUSPEND=0x00800080

ALARM_SEC=""

while [ $# -gt 0 ]; do
    case "$1" in
        --alarm)
            [ -z "$2" ] && { echo "ERRO: --alarm precisa do tempo em segundos" >&2; exit 1; }
            ALARM_SEC="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Uso:/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERRO: opcao desconhecida '$1' (use --help)" >&2
            exit 1 ;;
    esac
done

# Sem wake source: caminho classico req_shdn via pm_power_off.
if [ -z "$ALARM_SEC" ]; then
    devmem $EN_PWR_WAKEUP 32 0x00000000
    devmem $EN_PWR_VBAT_DET 32 0x00000000
    devmem $ALARM_ENABLE 32 0x00000000 2>/dev/null
    echo "[poweroff] req_shdn (off ate reconectar USB-C)"
    exec /sbin/poweroff
fi

# --- caminho suspend (--alarm) ---

if [ ! -e /sys/class/rtc/rtc0/wakealarm ]; then
    [ -f "$RTC_KO" ] && insmod "$RTC_KO" 2>/dev/null
fi
if [ ! -w /sys/class/rtc/rtc0/wakealarm ]; then
    echo "ERRO: /sys/class/rtc/rtc0/wakealarm indisponivel" >&2
    exit 1
fi
echo 0 > /sys/class/rtc/rtc0/wakealarm 2>/dev/null
if ! echo "+${ALARM_SEC}" > /sys/class/rtc/rtc0/wakealarm; then
    echo "ERRO: falha ao armar wakealarm em +${ALARM_SEC}s" >&2
    exit 1
fi

WAKE_HEX=$(printf '0x%08x' "$ALARM_BITS")

# Caminho oficial: kernel CONFIG_SUSPEND + SBI suspend.
if [ -e /sys/power/state ] && grep -q mem /sys/power/state 2>/dev/null; then
    echo "[suspend] caminho SBI oficial, alarme em ${ALARM_SEC}s"

    devmem $EN_PWR_VBAT_DET 32 0x00000000
    devmem $EN_PWR_WAKEUP 32 "$WAKE_HEX"

    # O aic8800 (mmc2:390b:2 SDIO) nao tem suspend handler funcional
    # e retorna -22 (-EINVAL) durante PM. Descarrega antes do suspend.
    SAVED_WIFI_MODE=""
    if lsmod 2>/dev/null | grep -q "^aic8800"; then
        if [ -f /mnt/system/wifi-lib.sh ]; then
            # shellcheck disable=SC1091
            . /mnt/system/wifi-lib.sh
            SAVED_WIFI_MODE=$(wifi_current_mode)
            wifi_stop_all 2>/dev/null
        else
            ifconfig wlan0 down 2>/dev/null
        fi
        rmmod aic8800_fdrv 2>/dev/null
        rmmod aic8800_bsp  2>/dev/null
        echo "[suspend] aic8800 descarregado (modo prev=$SAVED_WIFI_MODE)"
    fi

    sync
    if echo mem > /sys/power/state; then
        echo "[suspend] sistema acordou"
    else
        echo "[suspend] ERRO: echo mem falhou (algum driver bloqueou suspend?)"
        echo "[suspend] cheque dmesg | tail pra ver qual driver"
    fi

    # Recarrega aic8800 e restaura modo de Wi-Fi.
    if [ -n "$SAVED_WIFI_MODE" ] && [ -f /mnt/system/ko/aic8800_bsp.ko ]; then
        insmod /mnt/system/ko/aic8800_bsp.ko  2>/dev/null
        sleep 0.5
        insmod /mnt/system/ko/aic8800_fdrv.ko 2>/dev/null
        i=0
        while ! ip link show wlan0 >/dev/null 2>&1 && [ $i -lt 20 ]; do
            sleep 0.5; i=$((i+1))
        done
        case "$SAVED_WIFI_MODE" in
            ap)     wifi_start_ap     2>/dev/null && echo "[suspend] wifi AP restaurado" ;;
            client) wifi_start_client 2>/dev/null && echo "[suspend] wifi Client restaurado" ;;
        esac
    fi

    exit 0
fi

# Fallback: req_suspend manual (kernel sem CONFIG_SUSPEND)
echo "[suspend] fallback manual (kernel sem CONFIG_SUSPEND), alarme em ${ALARM_SEC}s"
echo "[suspend] AVISO: vai matar processos e cold-bootar ao acordar"

killall5 -15 2>/dev/null
sleep 2
killall5 -9 2>/dev/null
sleep 1

sync; sync; sync
swapoff -a 2>/dev/null
mount -o remount,ro / 2>/dev/null
mount -o remount,ro /mnt/data 2>/dev/null
mount -o remount,ro /mnt/sd 2>/dev/null
umount -a -r 2>/dev/null

devmem $EN_PWR_VBAT_DET 32 0x00000000
devmem $EN_PWR_WAKEUP 32 "$WAKE_HEX"

devmem $RTC_PRDATA_SEL 32 0x00000001
devmem $RTC_CTRL0_UNLOCK 32 0x0000AB18
devmem $RTC_EN_SUSPEND_REQ 32 0x00000001

while true; do
    devmem $RTC_CTRL0 32 $RTC_CTRL0_REQ_SUSPEND
    sleep 0.5
done
