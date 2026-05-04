#!/bin/sh
#
# node-red-services.sh - Start/stop Node-RED / SSCMA services and
#                        enable/disable them across reboots.
#
# Usage: node-red-services.sh {start|stop}
#
#   start : enables (renames K* -> S*) and starts the services
#   stop  : stops and disables (renames S* -> K*) the services
#
# Enable/disable persists by renaming the leading letter of the init
# script: BusyBox rcS only iterates /etc/init.d/S??* (start), and rcK
# iterates the same pattern with "stop". A K-prefixed file is invisible
# to both -- which is exactly what we want for a disabled service.
# (Note: K here is just a "disabled" marker; rcK does NOT auto-run K*.)
#

INITD="/etc/init.d"
# Order matters: services are listed in start-order. stop walks them in
# reverse so dependents go down before what they depend on.
SERVICES="03node-red 91sscma-node 93sscma-supervisor"

# Locate the script regardless of S/K prefix. Echoes the full path or
# nothing if neither exists.
find_script() {
    for prefix in S K; do
        if [ -f "$INITD/$prefix$1" ]; then
            echo "$INITD/$prefix$1"
            return
        fi
    done
}

case "$1" in
  start)
    echo "Enabling and starting Node-RED / SSCMA services..."
    for svc in $SERVICES; do
        script=$(find_script "$svc")
        if [ -z "$script" ]; then
            echo "  WARNING: neither S$svc nor K$svc found, skipping."
            continue
        fi
        case "$script" in
            */K*) mv "$script" "$INITD/S$svc"; script="$INITD/S$svc" ;;
        esac
        "$script" start
    done
    echo "Done."
    ;;

  stop)
    echo "Stopping and disabling Node-RED / SSCMA services..."
    REVERSED=""
    for svc in $SERVICES; do REVERSED="$svc $REVERSED"; done
    for svc in $REVERSED; do
        script=$(find_script "$svc")
        if [ -z "$script" ]; then
            echo "  WARNING: neither S$svc nor K$svc found, skipping."
            continue
        fi
        "$script" stop
        case "$script" in
            */S*) mv "$script" "$INITD/K$svc" ;;
        esac
    done
    echo "Done."
    ;;

  *)
    echo "Usage: $0 {start|stop}"
    exit 1
    ;;
esac

exit 0
