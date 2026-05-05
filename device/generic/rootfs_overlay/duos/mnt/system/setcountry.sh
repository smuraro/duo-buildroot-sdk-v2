#!/bin/sh
#
# /mnt/system/setcountry.sh - Persiste o country code regulatório do WiFi.
#
# CONTEXTO
# --------
# O driver AIC (chip WiFi do Duo S) e o cfg80211 do kernel obedecem a um
# domínio regulatório (canais permitidos, EIRP máximo, DFS, etc) baseado
# no country code ISO 3166-1 alpha-2 (ex: BR, US, DE, JP, GB).
#
# Single source of truth: /mnt/data/wifi-country (2 letras maiúsculas).
# Lido por:
#   - duo-init.sh        → `iw reg set` no kernel após carregar aic8800_fdrv
#   - wifi-lib.sh        → substitui `country_code=` no /tmp/hostapd.runtime.conf
#   - poweroff.sh        → reaplica após resume de suspend
#   - lowpower-restore   → idem após low-power
#
# Default: arquivo ausente/vazio → "00" (world domain, regras permissivas
# universais conservadoras).
#
# REGDB
# -----
# As regras vêm de /lib/firmware/regulatory.db (pacote wireless-regdb,
# assinado pela mantenedora wens). É um binário v20 onde cada país são
# 2 bytes ASCII crus — `strings` (min-len 4) não extrai e usar -n 2 vira
# lixo. Por isso este script não tenta listar/validar contra a regdb;
# confia na sigla ISO 3166-1 alpha-2 e deixa kernel/hostapd validarem.
#
# USO
# ---
#   setcountry.sh           → modo interativo (mostra estado, lista países,
#                             pergunta, persiste e reboota)
#   setcountry.sh BR        → modo direto: persiste BR; sem reboot automático
#   setcountry.sh clear     → remove arquivo (volta pra world domain "00")
#   setcountry.sh -h        → ajuda
#
# Mudança só vale após **reboot** — o `iw reg set` do kernel só roda no
# boot/resume; não há trigger runtime estável sem restart do hostapd.
#

WIFI_COUNTRY_FILE=/mnt/data/wifi-country
HOSTAPD_RUNTIME=/tmp/hostapd.runtime.conf

usage() {
    cat <<EOF
Uso: $0 [COUNTRY | clear | 00]
  $0              modo interativo (recomendado)
  $0 BR           persiste o país (2 letras ISO 3166-1 alpha-2); sem reboot auto
  $0 clear        remove o arquivo (volta pra "00" / world domain)
  $0 00           idem 'clear'

País fica em $WIFI_COUNTRY_FILE. Reboot pra mudança ter efeito.
EOF
}

valid_country() {
    case "$1" in
        [A-Z][A-Z]) return 0 ;;
        *)          return 1 ;;
    esac
}

show_state() {
    file_val=$(cat "$WIFI_COUNTRY_FILE" 2>/dev/null | tr -d '[:space:]' | tr a-z A-Z)
    kernel_val=$(iw reg get 2>/dev/null | awk '/^country/ {sub(/:/,"",$2); print $2; exit}')
    hostapd_val=$(awk -F= '/^country_code=/ {print $2; exit}' "$HOSTAPD_RUNTIME" 2>/dev/null)

    echo "Estado atual:"
    echo "  Arquivo  ($WIFI_COUNTRY_FILE):     ${file_val:-<vazio/ausente>}"
    echo "  Kernel   (cfg80211):                ${kernel_val:-<indisponível>}"
    echo "  hostapd  ($HOSTAPD_RUNTIME): ${hostapd_val:-<não rodando>}"

    # Diagnóstico:
    #  - arquivo vazio + kernel vazio/00 = world domain (sem ação)
    #  - arquivo vazio + kernel != 00    = clear pendente de reboot, NÃO é divergência
    #  - arquivo == kernel               = coerente
    #  - kernel vazio                    = wlan0/driver indisponível, sem como conferir
    #  - resto                           = divergência real (kernel ficou com setting estranho)
    if [ -z "$file_val" ]; then
        if [ -z "$kernel_val" ] || [ "$kernel_val" = "00" ]; then
            echo "  → world domain (\"00\") — default"
        else
            echo "  → arquivo vazio (default 00); kernel ainda em $kernel_val — reboot pendente"
        fi
    elif [ -z "$kernel_val" ]; then
        echo "  → kernel indisponível (driver/wlan0 ausente?); arquivo persiste $file_val"
    elif [ "$file_val" = "$kernel_val" ]; then
        echo "  → coerente"
    else
        echo "  → ATENÇÃO: arquivo=$file_val, kernel=$kernel_val divergem (reboot pra reconciliar)"
    fi
}

# Persiste o country code no arquivo. Não valida contra regdb (binário
# não extraível via `strings`). Kernel/hostapd validam no boot.
persist_country() {
    cc="$1"
    mkdir -p "$(dirname "$WIFI_COUNTRY_FILE")"
    echo "$cc" > "$WIFI_COUNTRY_FILE" && sync
    echo "Persistido: $WIFI_COUNTRY_FILE = $cc"
}

clear_country() {
    if [ -f "$WIFI_COUNTRY_FILE" ]; then
        rm -f "$WIFI_COUNTRY_FILE" && sync
        echo "Removido $WIFI_COUNTRY_FILE. Volta pra \"00\" (world domain) no próximo boot."
    else
        echo "$WIFI_COUNTRY_FILE já não existe (já está em \"00\")."
    fi
}

# Reboot com countdown de 3s pra dar chance de Ctrl-C.
do_reboot() {
    echo
    echo "Reboot em 3 segundos... (Ctrl-C pra cancelar)"
    sleep 1; echo "  3..."
    sleep 1; echo "  2..."
    sleep 1; echo "  1..."
    exec reboot
}

# Modo interativo: usado quando script é chamado sem argumentos.
interactive() {
    show_state
    echo
    echo "Códigos ISO 3166-1 alpha-2 (ex: BR, US, DE, JP, GB, FR, ES, IT, AR, CL, MX...)."
    echo "Lista completa: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2"
    echo
    echo "Digite:"
    echo "  - 2 letras (ex: BR, US, DE)         → persiste"
    echo "  - 'clear' ou '00'                   → volta pra world domain"
    echo "  - Enter (vazio)                     → cancela"
    echo
    printf "Country code: "
    read -r ans

    case "$ans" in
        "")
            echo "Cancelado, nenhuma mudança."
            exit 0 ;;
        clear|CLEAR|00)
            clear_country ;;
        *)
            cc=$(printf '%s' "$ans" | tr a-z A-Z)
            if ! valid_country "$cc"; then
                echo "ERRO: '$ans' inválido (esperado 2 letras maiúsculas, ex: BR)" >&2
                exit 1
            fi
            persist_country "$cc"
            ;;
    esac

    echo
    printf "Reboot agora pra aplicar? [S/n]: "
    read -r confirm
    case "$confirm" in
        n|N|no|NO|nao|NAO|"não"|"NÃO")
            echo "Reboot cancelado. Rode 'reboot' depois pra aplicar."
            exit 0 ;;
        *)
            do_reboot ;;
    esac
}

# ----- entry point -----
case "$1" in
  -h|--help)
    usage; exit 0 ;;

  "")
    interactive; exit 0 ;;

  clear|00)
    clear_country; exit 0 ;;
esac

# Argumento é um country code candidato.
cc=$(printf '%s' "$1" | tr a-z A-Z)
if ! valid_country "$cc"; then
    echo "ERRO: '$1' não é um country code válido (esperado 2 letras, ex: BR)" >&2
    usage >&2
    exit 1
fi
persist_country "$cc"
echo
echo "Reboot pra mudança valer no kernel e no hostapd."
