# ifndef CC
	# CC = g++
	# pybind11
CC=c++
# endif

# CCFLAGS= --shared -fPIC -O3 -std=c99 -I /Users/gerrie/anaconda3/envs/py36/include/python3.6m -std=c++11
CCFLAGS=  -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup  -fPIC `python3 -m pybind11 --includes`
LIBS = -lOpenCL -lm

COMMON_DIR = ../../C_common

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL -lm
endif

CCFLAGS += -D DEVICE=$(DEVICE)

nms: nms_opencl_vision.cpp
	# $(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@`python3-config --extension-suffix`
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o nms_opencl`python3-config --extension-suffix`


clean:
	rm -f nms *.o
