################################################################################
#
# ttyd
#
################################################################################

TTYD_VERSION = 1.7.7
TTYD_SITE = $(call github,tsl0922,ttyd,$(TTYD_VERSION))
TTYD_LICENSE = MIT
TTYD_LICENSE_FILES = LICENSE
TTYD_DEPENDENCIES = json-c libuv libwebsockets openssl zlib
TTYD_CPE_ID_VALID = YES

define TTYD_INSTALL_INIT_SYSV
	$(INSTALL) -D -m 0755 package/ttyd/S50ttyd \
		$(TARGET_DIR)/etc/init.d/S50ttyd
endef

$(eval $(cmake-package))
