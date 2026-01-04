# RingRift Sentinel - Build and Install
#
# Build:   make
# Install: sudo make install
# Test:    make test

CC = cc
CFLAGS = -Wall -Wextra -O2
TARGET = ringrift_sentinel
INSTALL_PATH = /usr/local/bin

.PHONY: all clean install uninstall test

all: $(TARGET)

$(TARGET): ringrift_sentinel.c
	$(CC) $(CFLAGS) -o $(TARGET) ringrift_sentinel.c

clean:
	rm -f $(TARGET)

install: $(TARGET)
	@echo "Installing $(TARGET) to $(INSTALL_PATH)..."
	install -m 755 $(TARGET) $(INSTALL_PATH)/$(TARGET)
	@echo "Done. Now load the launchd plist:"
	@echo "  sudo launchctl load /Library/LaunchDaemons/com.ringrift.sentinel.plist"

uninstall:
	@echo "Unloading launchd service..."
	-sudo launchctl unload /Library/LaunchDaemons/com.ringrift.sentinel.plist 2>/dev/null
	@echo "Removing $(TARGET)..."
	rm -f $(INSTALL_PATH)/$(TARGET)
	@echo "Done."

test: $(TARGET)
	@echo "Running sentinel in foreground (Ctrl+C to stop)..."
	./$(TARGET)
