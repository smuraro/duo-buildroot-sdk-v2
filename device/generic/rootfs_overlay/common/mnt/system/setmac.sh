#!/bin/sh
#
# /mnt/system/setmac.sh - Persist a stable eth0 MAC in U-Boot env.
#
# Por padrão o U-Boot gera um MAC random a cada boot e avisa "using
# random MAC address". Esse MAC volátil quebra DHCP reservation, ARP
# cache e ACLs por MAC.
#
# Esse script grava um MAC em `ethaddr` no env do U-Boot (mmcblk0p3).
# A partir do próximo boot o U-Boot lê o env e patcheia o DTB → o
# kernel sobe já com o MAC correto, sem warning.
#
# Uso:
#   setmac.sh                   # deriva do UID do chip (efuse)
#   setmac.sh AA:BB:CC:DD:EE:FF # usa o MAC informado
#   setmac.sh clear             # apaga o ethaddr (volta a random)
#   setmac.sh -h | --help
#
# Após rodar, é necessário reboot pra mudança ter efeito.
#

UID_FILE=/sys/class/cvi-base/base_uid

usage() {
    cat <<EOF
Uso: $0 [MAC | clear]
  $0                          deriva MAC do UID do chip ($UID_FILE)
  $0 AA:BB:CC:DD:EE:FF        usa o MAC informado
  $0 clear                    remove o ethaddr persistido (volta a random)

O MAC fica em ethaddr no env do U-Boot (mmcblk0p3). Reboot pra ter efeito.
EOF
}

valid_mac() {
    case "$1" in
        [0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F][0-9a-fA-F])
            return 0 ;;
        *)  return 1 ;;
    esac
}

case "$1" in
  -h|--help)
    usage; exit 0 ;;

  clear)
    echo "Removendo ethaddr do env..."
    if fw_setenv ethaddr; then
        echo "OK. Reboot pra voltar a random."
    else
        echo "ERRO: fw_setenv falhou" >&2
        exit 1
    fi
    exit 0 ;;
esac

if [ -n "$1" ]; then
    MAC=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
    if ! valid_mac "$MAC"; then
        echo "ERRO: '$1' não é um MAC válido (esperado XX:XX:XX:XX:XX:XX)" >&2
        usage >&2
        exit 1
    fi
else
    if [ ! -r "$UID_FILE" ]; then
        echo "ERRO: $UID_FILE indisponível." >&2
        echo "       O driver cvi-base sobe em S99user (final do boot)." >&2
        echo "       Aguarde o boot terminar e tente de novo." >&2
        exit 1
    fi
    # Formato esperado: "UID: XXXXXXXX_YYYYYYYY"
    HEX=$(awk '{print $2}' "$UID_FILE" | tr -d '_' | tr 'A-Z' 'a-z')
    if [ ${#HEX} -ne 16 ]; then
        echo "ERRO: formato inesperado em $UID_FILE: $(cat "$UID_FILE")" >&2
        exit 1
    fi
    # Últimos 10 hex chars = sufixo de 5 octetos. Prefixo 02 =
    # locally-administered, unicast.
    SFX=$(printf '%s' "$HEX" | tail -c 10)
    MAC=02:$(printf '%s' "$SFX" | sed 's/\(..\)\(..\)\(..\)\(..\)\(..\)/\1:\2:\3:\4:\5/')
fi

echo "Persistindo eth0 MAC=$MAC em ethaddr (mmcblk0p3)..."
if ! fw_setenv ethaddr "$MAC"; then
    echo "ERRO: fw_setenv falhou" >&2
    exit 1
fi

echo
fw_printenv ethaddr
echo
echo "Reboot pra mudança valer no eth0."
