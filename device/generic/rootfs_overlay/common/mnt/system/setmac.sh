#!/bin/sh
#
# /mnt/system/setmac.sh - Persiste MAC addresses estáveis pra eth0 e wlan0.
#
# CONTEXTO
# --------
# Por default ambas as interfaces sobem com MAC random:
#  - eth0: U-Boot avisa "using random MAC address" e gera um novo a cada
#    boot. Persistência via env var `ethaddr` (mmcblk0p3).
#  - wlan0: chip AIC com efuse não programado; firmware gera bytes 5-6
#    random a cada boot. Persistência via /mnt/data/wifi-mac (aplicado
#    em runtime por duo-init.sh com `ip link set wlan0 address`).
#
# Cada uma quebra DHCP reservation, ARP cache e MAC ACLs.
#
# USO
# ---
#   setmac.sh                 modo interativo (recomendado)
#   setmac.sh eth  [MAC]      eth0 (sem MAC = deriva do UID, prefixo 02)
#   setmac.sh wlan [MAC]      wlan0 (sem MAC = deriva do UID, prefixo 06)
#   setmac.sh all  [MAC]      ambos (sem MAC = deriva ambos do UID)
#                             com MAC: rejeita (use 2 chamadas separadas).
#   setmac.sh clear [eth|wlan|all]
#   setmac.sh -h
#
# Após mudar:
#   - eth0  vale após reboot (U-Boot patcheia DTB)
#   - wlan0 vale após reboot (duo-init.sh aplica via ip link)
#

UID_FILE=/sys/class/cvi-base/base_uid
WIFI_MAC_FILE=/mnt/data/wifi-mac

usage() {
    cat <<EOF
Uso: $0 [eth|wlan|all] [MAC] | clear [eth|wlan|all] | -h
  $0                       modo interativo
  $0 eth                   deriva eth0 do UID, prefixo 02
  $0 wlan                  deriva wlan0 do UID, prefixo 06
  $0 all                   deriva ambos (eth=02:..., wlan=06:...)
  $0 eth  AA:BB:..:FF      seta eth0 manualmente
  $0 wlan AA:BB:..:FF      seta wlan0 manualmente
  $0 clear all             remove ambos
  $0 clear eth             remove só eth0
  $0 clear wlan            remove só wlan0

Persistência:
  eth0  -> ethaddr no env do U-Boot (mmcblk0p3)
  wlan0 -> $WIFI_MAC_FILE (aplicado por duo-init.sh)

Mudanças valem só após reboot.
EOF
}

valid_mac() {
    case "$1" in
        [0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F])
            return 0 ;;
        *)  return 1 ;;
    esac
}

# Lê os 10 hex chars finais do UID do chip. Falha se driver cvi-base
# não estiver carregado (sobe em S99user — final do boot).
read_uid_suffix() {
    if [ ! -r "$UID_FILE" ]; then
        echo "ERRO: $UID_FILE indisponível (driver cvi-base sobe em S99user)." >&2
        return 1
    fi
    HEX=$(awk '{print $2}' "$UID_FILE" | tr -d '_' | tr 'A-Z' 'a-z')
    if [ ${#HEX} -ne 16 ]; then
        echo "ERRO: formato inesperado em $UID_FILE: $(cat "$UID_FILE")" >&2
        return 1
    fi
    printf '%s' "$HEX" | tail -c 10
}

# Monta MAC <prefix>:XX:XX:XX:XX:XX a partir dos 10 hex chars finais
# do UID. Prefix recomendado: 02 (eth) ou 06 (wlan) — locally-administered.
derive_mac() {
    prefix="$1"
    sfx=$(read_uid_suffix) || return 1
    printf '%s:%s\n' "$prefix" "$(printf '%s' "$sfx" | sed 's/\(..\)\(..\)\(..\)\(..\)\(..\)/\1:\2:\3:\4:\5/')"
}

# ----- ETH -----

set_eth() {
    mac=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
    if ! valid_mac "$mac"; then
        echo "ERRO: '$1' não é MAC válido" >&2
        return 1
    fi
    echo "eth0: persistindo MAC=$mac em ethaddr (mmcblk0p3)..."
    if ! fw_setenv ethaddr "$mac"; then
        echo "ERRO: fw_setenv falhou" >&2
        return 1
    fi
    fw_printenv ethaddr
}

clear_eth() {
    echo "eth0: removendo ethaddr do env..."
    if fw_setenv ethaddr; then
        echo "  OK. Reboot pra eth0 voltar a random."
    else
        echo "ERRO: fw_setenv falhou" >&2
        return 1
    fi
}

# ----- WLAN -----

set_wlan() {
    mac=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
    if ! valid_mac "$mac"; then
        echo "ERRO: '$1' não é MAC válido" >&2
        return 1
    fi
    echo "wlan0: persistindo MAC=$mac em $WIFI_MAC_FILE..."
    mkdir -p "$(dirname "$WIFI_MAC_FILE")"
    echo "$mac" > "$WIFI_MAC_FILE" && sync
    cat "$WIFI_MAC_FILE"
}

clear_wlan() {
    if [ -f "$WIFI_MAC_FILE" ]; then
        rm -f "$WIFI_MAC_FILE" && sync
        echo "wlan0: removido $WIFI_MAC_FILE."
        echo "  Reboot pra firmware voltar a gerar (random a cada boot)."
    else
        echo "wlan0: $WIFI_MAC_FILE já não existia."
    fi
}

# ----- INTERATIVO -----

show_state() {
    eth_env=$(fw_printenv -n ethaddr 2>/dev/null)
    eth_now=$(cat /sys/class/net/eth0/address 2>/dev/null)
    wlan_file=$(cat "$WIFI_MAC_FILE" 2>/dev/null | tr -d '[:space:]')
    wlan_now=$(cat /sys/class/net/wlan0/address 2>/dev/null)

    echo "Estado atual:"
    echo "  eth0  ethaddr (U-Boot env): ${eth_env:-<vazio = random>}"
    echo "  eth0  current /sys:         ${eth_now:-<down>}"
    echo "  wlan0 file ($WIFI_MAC_FILE): ${wlan_file:-<vazio = random>}"
    echo "  wlan0 current /sys:          ${wlan_now:-<down>}"
}

interactive() {
    show_state
    echo
    echo "O que fazer?"
    echo "  1) derivar e setar AMBOS (eth=02:..., wlan=06:..., do UID do chip)"
    echo "  2) só eth (deriva do UID)"
    echo "  3) só wlan (deriva do UID)"
    echo "  4) eth manual (informa MAC)"
    echo "  5) wlan manual (informa MAC)"
    echo "  6) clear ambos"
    echo "  7) clear eth"
    echo "  8) clear wlan"
    echo "  Enter = cancela"
    echo
    printf "Opção: "
    read -r opt
    case "$opt" in
        "")  echo "Cancelado."; exit 0 ;;
        1)
            mac_eth=$(derive_mac 02) || exit 1
            mac_wlan=$(derive_mac 06) || exit 1
            set_eth  "$mac_eth"
            set_wlan "$mac_wlan"
            ;;
        2)
            mac_eth=$(derive_mac 02) || exit 1
            set_eth  "$mac_eth"
            ;;
        3)
            mac_wlan=$(derive_mac 06) || exit 1
            set_wlan "$mac_wlan"
            ;;
        4)
            printf "MAC pra eth0 (XX:XX:XX:XX:XX:XX): "
            read -r m
            set_eth "$m" ;;
        5)
            printf "MAC pra wlan0 (XX:XX:XX:XX:XX:XX): "
            read -r m
            set_wlan "$m" ;;
        6)  clear_eth; clear_wlan ;;
        7)  clear_eth ;;
        8)  clear_wlan ;;
        *)  echo "Opção inválida."; exit 1 ;;
    esac

    echo
    printf "Reboot agora pra aplicar? [S/n]: "
    read -r confirm
    case "$confirm" in
        n|N|no|NO|nao|NAO|"não"|"NÃO")
            echo "Reboot cancelado. Rode 'reboot' depois pra aplicar."
            exit 0 ;;
        *)
            echo "Reboot em 3 segundos... (Ctrl-C pra cancelar)"
            sleep 1; echo "  3..."
            sleep 1; echo "  2..."
            sleep 1; echo "  1..."
            exec reboot ;;
    esac
}

# ----- ENTRY POINT -----

case "$1" in
  -h|--help)
    usage; exit 0 ;;

  "")
    interactive; exit 0 ;;

  clear)
    case "$2" in
        eth)         clear_eth ;;
        wlan)        clear_wlan ;;
        all|"")      clear_eth; clear_wlan ;;
        *) echo "ERRO: 'clear $2' inválido. Use clear [eth|wlan|all]" >&2; exit 1 ;;
    esac
    echo; echo "Reboot pra aplicar."
    exit 0 ;;

  eth)
    if [ -n "$2" ]; then
        set_eth "$2" || exit 1
    else
        m=$(derive_mac 02) || exit 1
        set_eth "$m" || exit 1
    fi
    echo; echo "Reboot pra aplicar."
    exit 0 ;;

  wlan)
    if [ -n "$2" ]; then
        set_wlan "$2" || exit 1
    else
        m=$(derive_mac 06) || exit 1
        set_wlan "$m" || exit 1
    fi
    echo; echo "Reboot pra aplicar."
    exit 0 ;;

  all)
    if [ -n "$2" ]; then
        echo "ERRO: 'all <MAC>' é ambíguo (não dá pra usar o mesmo MAC em 2 NICs)." >&2
        echo "       Use 'setmac.sh eth $2' e 'setmac.sh wlan <outro>' separadamente." >&2
        exit 1
    fi
    me=$(derive_mac 02) || exit 1
    mw=$(derive_mac 06) || exit 1
    set_eth  "$me" || exit 1
    set_wlan "$mw" || exit 1
    echo; echo "Reboot pra aplicar."
    exit 0 ;;

  *)
    echo "ERRO: comando desconhecido: $1" >&2
    usage >&2
    exit 1 ;;
esac
