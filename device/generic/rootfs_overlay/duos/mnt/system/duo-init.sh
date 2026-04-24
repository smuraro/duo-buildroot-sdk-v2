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
insmod /mnt/system/ko/aic8800_bsp.ko
sleep 0.5
insmod /mnt/system/ko/aic8800_fdrv.ko

# Insmod PWM Module
insmod /mnt/system/ko/cv181x_pwm.ko

# WiFi Access Point
(
	for i in $(seq 1 30); do
		ip link show wlan0 >/dev/null 2>&1 && break
		sleep 0.5
	done
	if ip link show wlan0 >/dev/null 2>&1; then
		killall -q wpa_supplicant 2>/dev/null
		ifconfig wlan0 192.168.50.1 netmask 255.255.255.0 up
		iw reg set BR 2>/dev/null
		hostapd -B /etc/hostapd.conf
		/etc/init.d/S80dnsmasq restart

		# NAT: clientes WiFi saem pela interface upstream (eth0/usb0/etc.)
		echo 1 > /proc/sys/net/ipv4/ip_forward
		iptables -t nat -A POSTROUTING ! -o wlan0 -j MASQUERADE
		iptables -A FORWARD -i wlan0 -j ACCEPT
		iptables -A FORWARD -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT
	fi
) &

