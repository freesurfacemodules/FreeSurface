RACK_DIR ?= ../..

FLAGS += -std=c++17 -g -Isrc/dep/include -Isrc/dep/DSPFilters/include
SOURCES += $(wildcard src/*.cpp)
SOURCES += $(wildcard src/dep/DSPFilters/source/*.cpp)
DISTRIBUTABLES += $(wildcard LICENSE*) res

include $(RACK_DIR)/plugin.mk

# Audinux's Rack build (Fedora aarch64) links libstdc++ dynamically and its
# plugins follow suit (the official plugin.mk links it statically), and its
# patched Rack loads user plugins from plain <user dir>/plugins (no
# -lin-<cpu> suffix; see rack-v2-0001-initialize-system-path.patch in the
# Audinux SRPM).  Both overrides apply only on aarch64 Linux.
PLUGINS_DIR := $(RACK_USER_DIR)/plugins-$(ARCH_OS)-$(ARCH_CPU)
ifdef ARCH_LIN
ifdef ARCH_ARM64
	LDFLAGS := $(filter-out -static-libstdc++ -static-libgcc,$(LDFLAGS))
	PLUGINS_DIR := $(RACK_USER_DIR)/plugins
endif
endif

# Install the unpacked plugin directory (Rack also extracts .vcvplugin
# packages from the plugins directory at startup; this is the direct form).
install-dir: dist
	mkdir -p "$(PLUGINS_DIR)"
	rm -rf "$(PLUGINS_DIR)/$(SLUG)"
	cp -r dist/$(SLUG) "$(PLUGINS_DIR)/"

# Stand-alone DSP self-test for the LG57 pipeline (no Rack dependency).
test: build/selftest
	./build/selftest

build/selftest: test/selftest.cpp $(wildcard src/dsp/*.hpp) $(wildcard src/generated/*.hpp)
	@mkdir -p build
	$(CXX) -std=c++17 -O3 -ffast-math -Isrc -o $@ $<

.PHONY: test install-dir
