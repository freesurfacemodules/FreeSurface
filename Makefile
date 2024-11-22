RACK_DIR ?= ../..

FLAGS += -g -Isrc/dep/include
SOURCES += $(wildcard src/*.cpp)
DISTRIBUTABLES += $(wildcard LICENSE*) res

include $(RACK_DIR)/plugin.mk
