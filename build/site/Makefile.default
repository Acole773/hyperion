# c
CC := gcc
GCCOPTS := -O3 -fno-math-errno -ffinite-math-only -frounding-math
CFLAGS := -std=c99 -Wall -Wextra --pedantic $(GCCOPTS)

# SIMD
CFLAGS_SIMD := -mavx512f -march=native -D__HYPERION_USE_SIMD

# linker ld 
LD := gcc # Use the gcc toolchain
LD_LIBS := -lm -L$(OBJECT_DIR) -l:libhyburn.a 
LD_FLAGS := 

# python
PYTHON := python3

# hip rocm
HIPCC := hipcc
HIPLD := hipcc
CFLAGS_GPU := 
LD_FLAGS_GPU :=
