#!/bin/sh
# Milkv Duo S poweroff/suspend wrapper.
#
# Uso:
#   poweroff.sh                       # req_shdn classico (off ate reconectar USB-C)
#   poweroff.sh --alarm SEC           # suspend, acorda em SEC segundos
#   poweroff.sh --alarm SEC --pin SPEC --wake-state high|low [--timeout SEC]
#                                     # polled wake: dorme SEC s, ao acordar le pino,
#                                     # se bate com wake-state sai; senao dorme de novo.
#                                     # Wifi para 1x antes do loop, restaura 1x ao sair.
#                                     # --timeout limita tempo total no loop (default 14400=4h, 0=inf).
#
# Formatos de --pin SPEC (auto-detectado):
#   H21       posicao fisica do header Duo S (1-50, mapa em pinpong/milkvDuo.py)
#   B17       pad/package: letra A-E = banco GPIO (porta..porte) + offset
#   4:7       gpiochip:offset (chip 0=porta, 4=porte)
#   462       sysfs raw (numero direto em /sys/class/gpio/gpioN)
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
#
#   --pin contorna isso via polled wake: RTC alarm acorda periodicamente,
#   userspace le GPIO, decide continuar dormindo ou sair do loop.
#   Pull-up/pull-down nao e configuravel pelo script (kernel pinctrl-cv181x
#   nao expoe pinconf). Use pull externo na placa ou pino driven pela fonte.

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

# Header position -> sysfs gpio (Duo S). Fonte:
# device/generic/rootfs_overlay/duos/usr/lib/python3.12/site-packages/pinpong/extension/milkvDuo.py
HEADER_MAP="3:468 5:469 7:466 8:496 10:497 11:459 12:467 13:460 15:470 16:500 18:499 19:461 21:462 22:498 23:463 24:464 26:508 27:429 28:431 29:428 30:430 33:433 34:437 35:432 36:436 39:435 40:352 41:434 42:353 44:354 46:451 48:450 50:449"

ALARM_SEC=""
PIN_SPEC=""
WAKE_STATE=""
TIMEOUT_SEC=14400  # 4h; 0 desabilita

while [ $# -gt 0 ]; do
    case "$1" in
        --alarm)
            [ -z "$2" ] && { echo "ERRO: --alarm precisa do tempo em segundos" >&2; exit 1; }
            ALARM_SEC="$2"; shift 2 ;;
        --pin)
            [ -z "$2" ] && { echo "ERRO: --pin precisa do especificador (H21|B17|4:7|462)" >&2; exit 1; }
            PIN_SPEC="$2"; shift 2 ;;
        --wake-state)
            [ -z "$2" ] && { echo "ERRO: --wake-state precisa de high|low" >&2; exit 1; }
            WAKE_STATE="$2"; shift 2 ;;
        --timeout)
            [ -z "$2" ] && { echo "ERRO: --timeout precisa do tempo em segundos (0=infinito)" >&2; exit 1; }
            TIMEOUT_SEC="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Uso:/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERRO: opcao desconhecida '$1' (use --help)" >&2
            exit 1 ;;
    esac
done

# --- helpers ---

# Resolve letra A-E para base sysfs do banco gpio correspondente.
# dwapb nao seta /sys/class/gpio/gpiochipN/label como "porta" etc; identifica
# pelo endereco do controlador no symlink do gpiochip (estavel via dts):
#   porta=0x03020000 portb=0x03021000 portc=0x03022000 portd=0x03023000 porte=0x05021000
resolve_bank_base() {
    local letter=$1 addr=""
    case "$letter" in
        a|A) addr=03020000 ;;
        b|B) addr=03021000 ;;
        c|C) addr=03022000 ;;
        d|D) addr=03023000 ;;
        e|E) addr=05021000 ;;
        *)   return 1 ;;
    esac
    local chip target
    for chip in /sys/class/gpio/gpiochip*; do
        [ -d "$chip" ] || continue
        target=$(readlink -f "$chip" 2>/dev/null)
        case "$target" in
            *${addr}.gpio*|*${addr}*)
                cat "$chip/base" 2>/dev/null && return 0 ;;
        esac
    done
    return 1
}

# Resolve --pin SPEC para numero sysfs absoluto. Detecta formato pelo prefixo.
resolve_pin_to_sysfs() {
    local spec=$1 pos off chip letter base
    case "$spec" in
        H[0-9]*)
            pos=${spec#H}
            for entry in $HEADER_MAP; do
                if [ "${entry%%:*}" = "$pos" ]; then
                    echo "${entry##*:}"
                    return 0
                fi
            done
            return 1 ;;
        [A-Ea-e][0-9]*)
            letter=$(echo "$spec" | cut -c1)
            off=$(echo "$spec" | cut -c2-)
            base=$(resolve_bank_base "$letter") || return 1
            echo $((base + off)) ;;
        [0-9]*:[0-9]*)
            chip=${spec%:*}
            off=${spec#*:}
            case "$chip" in
                0) letter=A ;;
                1) letter=B ;;
                2) letter=C ;;
                3) letter=D ;;
                4) letter=E ;;
                *) return 1 ;;
            esac
            base=$(resolve_bank_base "$letter") || return 1
            echo $((base + off)) ;;
        [0-9]*)
            echo "$spec" ;;
        *)
            return 1 ;;
    esac
}

# Normaliza wake-state -> 0|1.
wake_state_value() {
    case "$1" in
        high|HIGH|1) echo 1 ;;
        low|LOW|0)   echo 0 ;;
        *) return 1 ;;
    esac
}

# Le GPIO 3x com 1ms entre samples e retorna consenso (2-de-3). Em caso de
# leituras todas diferentes (impossivel, mas defesa em profundidade), retorna
# a 1a leitura.
read_gpio_debounced() {
    local sysfs=$1 v1 v2 v3 path="/sys/class/gpio/gpio$1/value"
    v1=$(cat "$path" 2>/dev/null)
    usleep 1000 2>/dev/null || sleep 0.001
    v2=$(cat "$path" 2>/dev/null)
    usleep 1000 2>/dev/null || sleep 0.001
    v3=$(cat "$path" 2>/dev/null)
    if [ "$v1" = "$v2" ]; then echo "$v1"
    elif [ "$v2" = "$v3" ]; then echo "$v2"
    elif [ "$v1" = "$v3" ]; then echo "$v1"
    else echo "$v1"
    fi
}

# Estado pra cleanup.
WIFI_WAS_LOADED=0
SAVED_WIFI_MODE=""
GPIO_EXPORTED_BY_US=""

# aic8800 (mmc2:390b:2 SDIO) nao tem suspend handler funcional e retorna
# -22 (-EINVAL) durante PM. Descarrega antes do suspend, recarrega depois.
stop_wifi() {
    if lsmod 2>/dev/null | grep -q "^aic8800"; then
        WIFI_WAS_LOADED=1
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
}

restore_wifi() {
    if [ "$WIFI_WAS_LOADED" = 1 ] && [ -n "$SAVED_WIFI_MODE" ] && [ -f /mnt/system/ko/aic8800_bsp.ko ]; then
        insmod /mnt/system/ko/aic8800_bsp.ko 2>/dev/null
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
}

# SIGINT/SIGTERM: restaura wifi e desexporta GPIO antes de sair.
cleanup_and_exit() {
    local rc=${1:-0}
    if [ -n "$GPIO_EXPORTED_BY_US" ]; then
        echo "$GPIO_EXPORTED_BY_US" > /sys/class/gpio/unexport 2>/dev/null
    fi
    restore_wifi
    exit "$rc"
}

# Validacao --pin / --wake-state.
if [ -n "$PIN_SPEC" ]; then
    if [ -z "$ALARM_SEC" ]; then
        echo "ERRO: --pin requer --alarm SEC (intervalo do polling)" >&2
        exit 1
    fi
    if [ -z "$WAKE_STATE" ]; then
        echo "ERRO: --pin requer --wake-state (high|low)" >&2
        exit 1
    fi
    if ! wake_state_value "$WAKE_STATE" >/dev/null; then
        echo "ERRO: --wake-state invalido '$WAKE_STATE' (use high|low)" >&2
        exit 1
    fi
fi

# --- caminho 1: req_shdn classico (sem --alarm) ---
if [ -z "$ALARM_SEC" ]; then
    devmem $EN_PWR_WAKEUP 32 0x00000000
    devmem $EN_PWR_VBAT_DET 32 0x00000000
    devmem $ALARM_ENABLE 32 0x00000000 2>/dev/null
    echo "[poweroff] req_shdn (off ate reconectar USB-C)"
    exec /sbin/poweroff
fi

# --- carrega RTC se necessario ---
if [ ! -e /sys/class/rtc/rtc0/wakealarm ]; then
    [ -f "$RTC_KO" ] && insmod "$RTC_KO" 2>/dev/null
fi
if [ ! -w /sys/class/rtc/rtc0/wakealarm ]; then
    echo "ERRO: /sys/class/rtc/rtc0/wakealarm indisponivel" >&2
    exit 1
fi

WAKE_HEX=$(printf '0x%08x' "$ALARM_BITS")

# --- caminho 2: SBI suspend (kernel CONFIG_SUSPEND) ---
if [ -e /sys/power/state ] && grep -q mem /sys/power/state 2>/dev/null; then

    # Setup de --pin: resolve sysfs, exporta como input. Se o pino ja estava
    # exportado por outro consumidor, NAO desexportamos no fim (so o nosso).
    SYSFS_PIN=""
    WAKE_VAL=""
    if [ -n "$PIN_SPEC" ]; then
        SYSFS_PIN=$(resolve_pin_to_sysfs "$PIN_SPEC") || {
            echo "ERRO: nao foi possivel resolver --pin '$PIN_SPEC'" >&2
            exit 1
        }
        WAKE_VAL=$(wake_state_value "$WAKE_STATE")

        if [ ! -d "/sys/class/gpio/gpio$SYSFS_PIN" ]; then
            if ! echo "$SYSFS_PIN" > /sys/class/gpio/export 2>/dev/null; then
                echo "ERRO: falha ao exportar GPIO $SYSFS_PIN" >&2
                exit 1
            fi
            GPIO_EXPORTED_BY_US=$SYSFS_PIN
        fi
        echo in > "/sys/class/gpio/gpio$SYSFS_PIN/direction" 2>/dev/null || {
            echo "ERRO: falha ao setar direction=in em GPIO $SYSFS_PIN" >&2
            cleanup_and_exit 1
        }

        echo "[wake-on-pin] pin=$PIN_SPEC sysfs=gpio$SYSFS_PIN wake-state=$WAKE_STATE($WAKE_VAL) alarm=${ALARM_SEC}s timeout=${TIMEOUT_SEC}s"
    fi

    # NAO zerar EN_PWR_VBAT_DET aqui: VBAT_DET nao dispara wake de ST_SUSP
    # (so de ST_OFF), e o valor permaneceria em 0 apos o resume — desabilitando
    # o pino fisico de reset (PWR_VBAT_DET) ate o proximo cold-boot.
    devmem $EN_PWR_WAKEUP 32 "$WAKE_HEX"

    # Wifi para 1x ANTES do loop e restaura 1x DEPOIS — evita re-cycling
    # caro a cada wake-falso.
    stop_wifi

    # Trap so depois que wifi parou e gpio exportou, pra cleanup ter o que fazer.
    trap 'cleanup_and_exit 130' INT
    trap 'cleanup_and_exit 143' TERM

    if [ -z "$PIN_SPEC" ]; then
        # --- 1 ciclo (comportamento original) ---
        echo 0 > /sys/class/rtc/rtc0/wakealarm 2>/dev/null
        if ! echo "+${ALARM_SEC}" > /sys/class/rtc/rtc0/wakealarm; then
            echo "ERRO: falha ao armar wakealarm em +${ALARM_SEC}s" >&2
            cleanup_and_exit 1
        fi
        echo "[suspend] caminho SBI oficial, alarme em ${ALARM_SEC}s"
        sync
        if echo mem > /sys/power/state; then
            echo "[suspend] sistema acordou"
            # FSBL bl2_main.c chama set_rtc_en_registers() ANTES de
            # jump_to_warmboot_entry, que limpa bit 2 do EN_PWR_VBAT_DET.
            # Sem bit 2, o pino fisico de reset (PWR_VBAT_DET) nao dispara
            # cold-boot ao ser aterrado — em vez disso o sistema congela.
            # Restaura bit 2 aqui (em ST_ON nao causa power-up espurio).
            devmem 0x050260D0 32 0x00000007
        else
            echo "[suspend] ERRO: echo mem falhou (algum driver bloqueou suspend?)"
            echo "[suspend] cheque dmesg | tail pra ver qual driver"
        fi
    else
        # --- polling loop (--pin) ---
        START_TIME=$(date +%s)
        ITER=0
        EXIT_REASON=""
        while true; do
            ITER=$((ITER + 1))

            # Re-arma alarm a cada iteracao (alarm e one-shot).
            echo 0 > /sys/class/rtc/rtc0/wakealarm 2>/dev/null
            if ! echo "+${ALARM_SEC}" > /sys/class/rtc/rtc0/wakealarm; then
                echo "[wake-on-pin] iter=$ITER ERRO: falha ao re-armar wakealarm" >&2
                EXIT_REASON="wakealarm-fail"
                break
            fi

            sync
            if ! echo mem > /sys/power/state; then
                echo "[wake-on-pin] iter=$ITER ERRO: echo mem falhou" >&2
                EXIT_REASON="suspend-fail"
                break
            fi
            # FSBL warmboot limpa bit 2 do EN_PWR_VBAT_DET — restaura aqui
            # pra manter o pino fisico de reset funcional entre iteracoes.
            devmem 0x050260D0 32 0x00000007 2>/dev/null

            PIN_VAL=$(read_gpio_debounced "$SYSFS_PIN")

            if [ "$PIN_VAL" = "$WAKE_VAL" ]; then
                echo "[wake-on-pin] iter=$ITER pino=$PIN_VAL bate (esperado=$WAKE_VAL), saindo"
                EXIT_REASON="match"
                break
            fi

            if [ "$TIMEOUT_SEC" -gt 0 ]; then
                NOW=$(date +%s)
                ELAPSED=$((NOW - START_TIME))
                if [ "$ELAPSED" -ge "$TIMEOUT_SEC" ]; then
                    echo "[wake-on-pin] iter=$ITER timeout ${TIMEOUT_SEC}s atingido (pino=$PIN_VAL, esperado=$WAKE_VAL), saindo"
                    EXIT_REASON="timeout"
                    break
                fi
            fi
        done
        echo "[wake-on-pin] terminou: $EXIT_REASON apos $ITER iter(s)"
    fi

    restore_wifi
    trap - INT TERM

    if [ -n "$GPIO_EXPORTED_BY_US" ]; then
        echo "$GPIO_EXPORTED_BY_US" > /sys/class/gpio/unexport 2>/dev/null
    fi
    exit 0
fi

# --- caminho 3: fallback req_suspend manual (kernel sem CONFIG_SUSPEND) ---

if [ -n "$PIN_SPEC" ]; then
    echo "ERRO: --pin requer kernel com CONFIG_SUSPEND (echo mem indisponivel)" >&2
    echo "ERRO: caminho fallback faz cold-boot ao acordar, incompativel com loop polled" >&2
    exit 1
fi

echo 0 > /sys/class/rtc/rtc0/wakealarm 2>/dev/null
if ! echo "+${ALARM_SEC}" > /sys/class/rtc/rtc0/wakealarm; then
    echo "ERRO: falha ao armar wakealarm em +${ALARM_SEC}s" >&2
    exit 1
fi

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

# NAO zerar EN_PWR_VBAT_DET (mesmo raciocinio do caminho oficial acima).
devmem $EN_PWR_WAKEUP 32 "$WAKE_HEX"

devmem $RTC_PRDATA_SEL 32 0x00000001
devmem $RTC_CTRL0_UNLOCK 32 0x0000AB18
devmem $RTC_EN_SUSPEND_REQ 32 0x00000001

while true; do
    devmem $RTC_CTRL0 32 $RTC_CTRL0_REQ_SUSPEND
    sleep 0.5
done
