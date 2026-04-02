#!/bin/sh
#
# node-red-services.sh - Start or stop Node-RED / SSCMA services and
#                        enable/disable them across reboots.
#
# Usage: node-red-services.sh {start|stop}
#
#   start : enables and starts  S03node-red, S91sscma-node, S93sscma-supervisor
#   stop  : disables and stops  S03node-red, S91sscma-node, S93sscma-supervisor
#
# "Enable/disable across reboots" is done by toggling the executable bit
# on the init.d scripts. BusyBox rcS only runs scripts that are executable.
#

INITD="/etc/init.d"
SERVICES="S03node-red S91sscma-node S93sscma-supervisor"

case "$1" in
  start)
    echo "Enabling and starting Node-RED / SSCMA services..."
    for svc in $SERVICES; do
      script="$INITD/$svc"
      if [ -f "$script" ]; then
        chmod +x "$script"
        "$script" start
      else
        echo "  WARNING: $script not found, skipping."
      fi
    done
    echo "Done."
    ;;

  stop)
    echo "Stopping and disabling Node-RED / SSCMA services..."
    for svc in $SERVICES; do
      script="$INITD/$svc"
      if [ -f "$script" ]; then
        "$script" stop
        chmod -x "$script"
      else
        echo "  WARNING: $script not found, skipping."
      fi
    done
    echo "Done."
    ;;

  *)
    echo "Usage: $0 {start|stop}"
    exit 1
    ;;
esac

exit 0
