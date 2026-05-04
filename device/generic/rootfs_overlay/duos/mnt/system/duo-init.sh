#!/bin/sh

function set_gpio()
{
	local gpio_num=$1
	local gpio_val=$2
	local gpio_path="/sys/class/gpio/gpio${gpio_num}"

	if test -d ${gpio_path}; then
		echo "GPIO ${gpio_num} already exported" >> /tmp/gpio.log 2>&1
	else
		echo ${gpio_num} > /sys/class/gpio/export
	fi

	echo out > ${gpio_path}/direction
	sleep 0.1
	echo ${gpio_val} > ${gpio_path}/value
}

# Hardware V1.1
gpio_b17=465
set_gpio ${gpio_b17} 0

# Host Wake BT
host_wake_bt=362
set_gpio ${host_wake_bt} 1

# WIFI/BT Module
#
# Country code regulatório:
#   - Lido de /mnt/data/wifi-country (single source of truth, ver wifi-lib.sh).
#   - Default 00 = world domain.
#   - Para mudar:  /mnt/system/setcountry.sh BR  &&  reboot
#
# custregd=0 desliga o modo "self-managed" do driver AIC. Sem isso o
# driver registra rules permissivas próprias e ignora cfg80211 — o que
# gera o warning "*** USING PERMISSIVE CUSTOM REGULATORY RULES ***" no
# kernel. Com custregd=0, a regdb (/lib/firmware/regulatory.db, do
# pacote wireless-regdb) e o `iw reg set` abaixo passam a valer.
. /mnt/system/wifi-lib.sh

insmod /mnt/system/ko/aic8800_bsp.ko
sleep 0.5
insmod /mnt/system/ko/aic8800_fdrv.ko custregd=0
sleep 0.5
# Aplica MAC persistido (se houver) — antes de qualquer cliente da rede
# ver wlan0. Helper espera o netdev aparecer e faz down/set/up.
wifi_apply_mac
# cfg80211 tenta carregar a regdb antes do rootfs montar (~1.8s vs 2.0s)
# e marca regdb=ERR_PTR(-ENODATA) sem retry. `iw reg reload` força um
# request_firmware síncrono que substitui o err pointer pelo regdb real.
iw reg reload 2>/dev/null
iw reg set "$(wifi_country)" 2>/dev/null

# Insmod PWM Module
insmod /mnt/system/ko/cv181x_pwm.ko

# Wi-Fi: decide AP vs Client conforme presença de /mnt/data/wpa_supplicant.conf.
# - Sem config válida → sobe AP imediatamente (SSID único por MAC).
# - Com config       → sobe Client + watchdog. Se não associar em N segundos
#                      ou cair por > grace minutos, watchdog volta pra AP.
(
	. /mnt/system/wifi-lib.sh

	# wlan0 só aparece depois que aic8800_fdrv carrega — espera até 15s.
	i=0
	while ! ip link show $WIFI_IFACE >/dev/null 2>&1 && [ $i -lt 30 ]; do
		sleep 0.5
		i=$((i+1))
	done
	ip link show $WIFI_IFACE >/dev/null 2>&1 || exit 0

	if wifi_client_config_ok; then
		if wifi_start_client; then
			# Watchdog em background — fallback automático pra AP.
			setsid /mnt/system/wifi-watchdog </dev/null >/dev/null 2>&1 &
		else
			# Falha imediata na subida do supplicant: cai já pra AP.
			wifi_start_ap
		fi
	else
		wifi_start_ap
	fi
) &

