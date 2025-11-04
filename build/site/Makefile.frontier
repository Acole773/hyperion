# Use Cray compiler wrappers for correct MPI and math linkage
CC := cc
LD := cc
PYTHON := python3
GCCOPTS := -O3 -fno-math-errno -ffinite-math-only -frounding-math
CFLAGS := -std=c99 -Wall -Wextra --pedantic $(GCCOPTS)
# NOTE: this must not be := defined, as it has to reeval CFLAGS for SIZE def
CFLAGS_SIMD = $(CFLAGS) -mavx512f \
							 -march=native -D__HYPERION_USE_SIND

# === HDF5 (Cray MPI build) ===
HDF5_DIR := /sw/frontier/spack-envs/cpe24.11-cpu/opt/cce-18.0.1/hdf5-1.14.5-fbhfdr23kwzgknnufevewoslxgxfbvo7
HDF5_INC := -I$(HDF5_DIR)/include
HDF5_LIB := -L$(HDF5_DIR)/lib

# Link against HDF5 and standard math/dependency libs
LD_LIBS := $(HDF5_LIB) -lhdf5 -lz -ldl -lm -lc -L$(OBJECT_DIR) -l:libhyburn.a
LD_FLAGS := $(HDF5_INC)
