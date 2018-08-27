CPP=g++
BUILD_DIR=./build
BINARY=$(BUILD_DIR)/CNN
SRC=src/CNN.cpp src/SAXHandler.cpp
INCLUDES=./include
LIBS=-lpthread -lxerces-c
OPENCV_INCLUDES_AND_LIBS=`pkg-config --cflags --libs opencv`
RM=rm -rf
FIND=find -name
PIPE=xargs
MKDIR=mkdir

all: clean create-build-dir debug

debug:
	$(CPP) -DDEBUG -o $(BINARY) $(SRC) -I$(INCLUDES) $(OPENCV_INCLUDES_AND_LIBS) $(LIBS)

nodebug:
	$(CPP) -o $(BINARY) $(SRC) -I$(INCLUDES) $(OPENCV_INCLUDES_AND_LIBS) $(LIBS)

clean:
	$(FIND) "*~" | $(PIPE) $(RM)
	$(RM) $(BUILD_DIR)

create-build-dir:
	$(MKDIR) $(BUILD_DIR)

run: run-sequential

run-sequential:
	$(BINARY) ./xml/cnn-config.xml features.dat 0

run-parallel:
	$(BINARY) ./xml/cnn-config.xml features.dat 1 10
