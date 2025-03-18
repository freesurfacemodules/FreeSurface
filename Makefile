RACK_DIR ?= ../..

FLAGS += -std=c++17 -g -Isrc/dep/include -Isrc/dep/DSPFilters/include
SOURCES += $(wildcard src/*.cpp)
SOURCES += $(wildcard src/dep/DSPFilters/source/*.cpp)
DISTRIBUTABLES += $(wildcard LICENSE*) res

include $(RACK_DIR)/plugin.mk
