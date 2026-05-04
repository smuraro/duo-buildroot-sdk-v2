#!/bin/sh
# Helpers de Wi-Fi do Duo S. Source-only (não executar direto).
#
# Cobre:
#  - geração do hostapd.conf de runtime com SSID único por device (MAC)
#  - start/stop ordenado de AP e Client
#  - persistência: presença de $WPA_SUPPLICANT_CONF decide o boot
#
# Estado de runtime:
#  /var/run/wifi_mode           -> "ap" | "client" | "off"
#  /var/run/hostapd.pid         -> PID do hostapd (-P)
#  /var/run/wpa_supplicant.pid  -> PID do wpa_supplicant (-P)
#  /var/run/udhcpc.wlan0.pid    -> PID do udhcpc (-p)
#  /var/run/wifi-watchdog.pid   -> PID do watchdog (auto-fallback)

WIFI_IFACE=wlan0
AP_IP=192.168.50.1
AP_NETMASK=255.255.255.0

HOSTAPD_TEMPLATE=/etc/hostapd.conf
HOSTAPD_RUNTIME=/tmp/hostapd.runtime.conf
HOSTAPD_PIDFILE=/var/run/hostapd.pid

# Caminho canônico do wpa_supplicant. O rootfs já vem com um placeholder
# (ssid="SSID"/psk="PASSWORD") shipado pelo pacote buildroot — wifi_client_config_ok
# rejeita esse placeholder pra evitar que o boot tente client com lixo.
WPA_SUPPLICANT_CONF=/etc/wpa_supplicant.conf
WPA_PIDFILE=/var/run/wpa_supplicant.pid
UDHCPC_PIDFILE=/var/run/udhcpc.wlan0.pid

WIFI_MODE_FLAG=/var/run/wifi_mode
WATCHDOG_PIDFILE=/var/run/wifi-watchdog.pid

# Overrides opcionais em /mnt/data/wifi.conf (key=value):
#   AP_SSID_PREFIX=MeuDuo
#   AP_PASSPHRASE=segredo123
#   CLIENT_CONNECT_TIMEOUT=60
#   CLIENT_DISCONNECT_GRACE=300
WIFI_OVERRIDES=/mnt/data/wifi.conf
AP_SSID_PREFIX_DEFAULT=DuoS-AP
CLIENT_CONNECT_TIMEOUT_DEFAULT=60
CLIENT_DISCONNECT_GRACE_DEFAULT=300

# Country code regulatório. Single source of truth: lido por duo-init.sh
# (pra `iw reg set` no kernel) e por wifi_render_hostapd_conf (pra
# `country_code=` do hostapd). Default 00 = world domain.
# Helper de gerência: /mnt/system/setcountry.sh (set/clear/inspect).
WIFI_COUNTRY_FILE=/mnt/data/wifi-country

wifi_country() {
    cc=$(cat "$WIFI_COUNTRY_FILE" 2>/dev/null | tr -d '[:space:]' | tr a-z A-Z)
    echo "${cc:-00}"
}

# MAC persistente pro wlan0. O firmware do AIC gera bytes 5-6 random
# a cada boot (efuse interno do chip não tem MAC programado), então
# persistimos via /mnt/data/wifi-mac e aplicamos com `ip link set`.
# Helper de gerência: /mnt/system/setmac.sh wlan ...
WIFI_MAC_FILE=/mnt/data/wifi-mac

# Lê e normaliza pra lowercase XX:XX:XX:XX:XX:XX, ou imprime nada.
wifi_mac() {
    cat "$WIFI_MAC_FILE" 2>/dev/null | tr -d '[:space:]' | tr A-Z a-z
}

# Aplica o MAC persistido em wlan0. Espera até 10s pelo netdev aparecer
# (driver SDIO é async). No-op se arquivo ausente. Idempotente —
# trazer wlan0 down/up não causa problema mesmo se já estava up.
wifi_apply_mac() {
    mac=$(wifi_mac)
    [ -z "$mac" ] && return 0
    i=0
    while [ ! -e /sys/class/net/wlan0 ] && [ $i -lt 20 ]; do
        sleep 0.5; i=$((i+1))
    done
    [ -e /sys/class/net/wlan0 ] || return 1
    ip link set wlan0 down 2>/dev/null
    ip link set wlan0 address "$mac" 2>/dev/null
    ip link set wlan0 up   2>/dev/null
}

wifi_log() {
    # 1 linha pra console + dmesg (visível no `dmesg | grep wifi`).
    echo "[wifi] $*"
    [ -w /dev/kmsg ] && echo "<6>wifi: $*" > /dev/kmsg 2>/dev/null
}

wifi_load_overrides() {
    AP_SSID_PREFIX="$AP_SSID_PREFIX_DEFAULT"
    AP_PASSPHRASE=""
    CLIENT_CONNECT_TIMEOUT="$CLIENT_CONNECT_TIMEOUT_DEFAULT"
    CLIENT_DISCONNECT_GRACE="$CLIENT_DISCONNECT_GRACE_DEFAULT"
    if [ -f "$WIFI_OVERRIDES" ]; then
        # eval só de linhas KEY=VALUE simples; ignora comentários.
        while IFS= read -r line; do
            case "$line" in
                ''|\#*) ;;
                AP_SSID_PREFIX=*|AP_PASSPHRASE=*|CLIENT_CONNECT_TIMEOUT=*|CLIENT_DISCONNECT_GRACE=*)
                    eval "$line" ;;
            esac
        done < "$WIFI_OVERRIDES"
    fi
}

# Sufixo único de 6 hex (3 últimos octetos do MAC do wlan0).
# Mantém estabilidade entre reboots (o aic8800 não randomiza MAC).
wifi_unique_suffix() {
    mac=$(cat /sys/class/net/$WIFI_IFACE/address 2>/dev/null)
    if [ -z "$mac" ]; then
        echo "UNKNOWN"
        return
    fi
    # aa:bb:cc:dd:ee:ff -> DDEEFF
    echo "$mac" | awk -F: '{ printf "%s%s%s\n", toupper($4), toupper($5), toupper($6) }'
}

wifi_render_hostapd_conf() {
    wifi_load_overrides
    [ -f "$HOSTAPD_TEMPLATE" ] || {
        wifi_log "ERRO: $HOSTAPD_TEMPLATE nao encontrado"
        return 1
    }
    suffix=$(wifi_unique_suffix)
    ssid="${AP_SSID_PREFIX}-${suffix}"
    country=$(wifi_country)
    # hostapd só aceita country codes ISO reais (BR, US, ...). Trata "00"
    # (world domain do cfg80211) como "sem país" — omite a linha. Sem ela
    # hostapd não força COUNTRY_UPDATE e respeita o que o kernel tem.
    [ "$country" = "00" ] && country=""

    # Substitui ssid=, country_code= (se houver country) e, se
    # AP_PASSPHRASE setado, wpa_passphrase=. country_code definido
    # precisa bater com o `iw reg set` do duo-init.sh, senão hostapd
    # sobrescreve o que o kernel tinha via COUNTRY_UPDATE.
    awk -v ssid="$ssid" -v pass="$AP_PASSPHRASE" -v country="$country" '
        BEGIN { saw_ssid=0; saw_pass=0; saw_cc=0 }
        /^[[:space:]]*ssid[[:space:]]*=/        { print "ssid=" ssid; saw_ssid=1; next }
        /^[[:space:]]*wpa_passphrase[[:space:]]*=/ {
            if (pass != "") { print "wpa_passphrase=" pass } else { print }
            saw_pass=1; next
        }
        /^[[:space:]]*country_code[[:space:]]*=/ {
            if (country != "") print "country_code=" country
            saw_cc=1; next
        }
        { print }
        END {
            if (!saw_ssid) print "ssid=" ssid
            if (!saw_pass && pass != "") print "wpa_passphrase=" pass
            if (!saw_cc && country != "") print "country_code=" country
        }
    ' "$HOSTAPD_TEMPLATE" > "$HOSTAPD_RUNTIME"
    wifi_log "AP SSID=$ssid (template=$HOSTAPD_TEMPLATE runtime=$HOSTAPD_RUNTIME)"
}

# Mata um daemon pelo seu pidfile, com graceful + force.
_wifi_kill_pidfile() {
    pf="$1"
    [ -f "$pf" ] || return 0
    pid=$(cat "$pf" 2>/dev/null)
    [ -z "$pid" ] || [ "$pid" -eq 0 ] 2>/dev/null && { rm -f "$pf"; return 0; }
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null
        i=0
        while kill -0 "$pid" 2>/dev/null && [ $i -lt 5 ]; do
            sleep 1; i=$((i+1))
        done
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
    fi
    rm -f "$pf"
}

wifi_stop_ap() {
    _wifi_kill_pidfile "$HOSTAPD_PIDFILE"
    # Limpa NAT (idempotente; ignora se a regra não existe).
    iptables -t nat -D POSTROUTING ! -o $WIFI_IFACE -j MASQUERADE 2>/dev/null
    iptables -D FORWARD -i $WIFI_IFACE -j ACCEPT 2>/dev/null
    iptables -D FORWARD -o $WIFI_IFACE -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null
}

wifi_stop_client() {
    _wifi_kill_pidfile "$UDHCPC_PIDFILE"
    _wifi_kill_pidfile "$WPA_PIDFILE"
}

wifi_stop_watchdog() {
    _wifi_kill_pidfile "$WATCHDOG_PIDFILE"
}

wifi_iface_down() {
    ip addr flush dev $WIFI_IFACE 2>/dev/null
    ifconfig $WIFI_IFACE down 2>/dev/null
}

wifi_stop_all() {
    wifi_stop_watchdog
    wifi_stop_ap
    wifi_stop_client
    wifi_iface_down
    echo off > "$WIFI_MODE_FLAG"
}

# Sobe AP. Idempotente: para hostapd/NAT/cliente prévios antes de subir.
wifi_start_ap() {
    [ -e /sys/class/net/$WIFI_IFACE ] || {
        wifi_log "ERRO: $WIFI_IFACE ausente; modulo aic8800 carregado?"
        return 1
    }
    wifi_stop_client
    wifi_stop_ap
    wifi_render_hostapd_conf || return 1
    iw reg set BR 2>/dev/null
    ifconfig $WIFI_IFACE $AP_IP netmask $AP_NETMASK up || return 1
    hostapd -B -P "$HOSTAPD_PIDFILE" "$HOSTAPD_RUNTIME" || {
        wifi_log "ERRO: hostapd falhou ao iniciar"
        return 1
    }
    /etc/init.d/S80dnsmasq restart >/dev/null 2>&1
    echo 1 > /proc/sys/net/ipv4/ip_forward
    iptables -t nat -A POSTROUTING ! -o $WIFI_IFACE -j MASQUERADE
    iptables -A FORWARD -i $WIFI_IFACE -j ACCEPT
    iptables -A FORWARD -o $WIFI_IFACE -m state --state RELATED,ESTABLISHED -j ACCEPT
    echo ap > "$WIFI_MODE_FLAG"
    wifi_log "modo AP ativo em $AP_IP"
}

# Valida config de cliente: existe, tem ssid=..., e NÃO é o placeholder
# default do pacote buildroot (ssid="SSID"/psk="PASSWORD") nem strings vazias.
wifi_client_config_ok() {
    [ -f "$WPA_SUPPLICANT_CONF" ] || return 1
    grep -q "^[[:space:]]*ssid=" "$WPA_SUPPLICANT_CONF" || return 1
    # Rejeita placeholders comuns: ssid="SSID", ssid="", psk="PASSWORD", psk="".
    awk '
        /^[[:space:]]*ssid[[:space:]]*=/ {
            v=$0; sub(/^[^=]*=[[:space:]]*/,"",v); gsub(/[ \t\r"]/,"",v)
            if (v=="" || v=="SSID") bad=1
        }
        /^[[:space:]]*psk[[:space:]]*=/ {
            v=$0; sub(/^[^=]*=[[:space:]]*/,"",v); gsub(/[ \t\r"]/,"",v)
            if (v=="" || v=="PASSWORD") bad=1
        }
        END { exit bad }
    ' "$WPA_SUPPLICANT_CONF"
}

# Sobe Client (sem watchdog). Falha se config inválida.
wifi_start_client() {
    [ -e /sys/class/net/$WIFI_IFACE ] || {
        wifi_log "ERRO: $WIFI_IFACE ausente"
        return 1
    }
    wifi_client_config_ok || {
        wifi_log "ERRO: $WPA_SUPPLICANT_CONF ausente ou invalido"
        return 1
    }
    wifi_stop_ap
    iw reg set BR 2>/dev/null
    ip addr flush dev $WIFI_IFACE 2>/dev/null
    ifconfig $WIFI_IFACE up || return 1
    wpa_supplicant -B -i $WIFI_IFACE -c "$WPA_SUPPLICANT_CONF" \
                   -P "$WPA_PIDFILE" -Dnl80211 || {
        wifi_log "ERRO: wpa_supplicant falhou"
        return 1
    }
    # udhcpc detachado: sem associacao no link, ele bloquearia ate -t*-T
    # segundos antes de virar daemon (~15s). Rodando em background com
    # setsid, wifi_start_client retorna imediato e o udhcpc continua
    # tentando ate o wpa_supplicant associar. Watchdog cobre o caso de
    # nunca pegar IP.
    setsid udhcpc -i $WIFI_IFACE -p "$UDHCPC_PIDFILE" -R -b -t 5 -T 3 \
        </dev/null >/dev/null 2>&1 &
    echo client > "$WIFI_MODE_FLAG"
    wifi_log "modo Client iniciado (config=$WPA_SUPPLICANT_CONF)"
}

# True se wlan0 está associado a um AP.
wifi_client_associated() {
    state=$(wpa_cli -i $WIFI_IFACE status 2>/dev/null | \
            awk -F= '$1=="wpa_state"{print $2}')
    [ "$state" = "COMPLETED" ]
}

# True se wlan0 tem IP configurado.
wifi_client_has_ip() {
    ip -4 addr show dev $WIFI_IFACE 2>/dev/null | grep -q "inet "
}

wifi_current_mode() {
    [ -f "$WIFI_MODE_FLAG" ] && cat "$WIFI_MODE_FLAG" || echo off
}
