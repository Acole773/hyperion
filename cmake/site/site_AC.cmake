# ------------------------------------------------------------
# Adam Cole local build settings (CMake version)
# ------------------------------------------------------------

# Compiler
set(CMAKE_C_COMPILER gcc)

# Optimization flags
set(GCCOPTS "-O3 -fno-math-errno -ffinite-math-only -frounding-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra --pedantic ${GCCOPTS}")

# SIMD options
set(CFLAGS_SIMD "${CMAKE_C_FLAGS} -mavx512f -march=native -D__HYPERION_USE_SIMD")

# HDF5 include + link paths
set(HDF5_INCLUDE_DIR "/usr/include/hdf5/serial")
set(HDF5_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu/hdf5/serial")

include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIR})

# Static HDF5 libs
set(HDF5_STATIC
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a
)

# Extra dependencies
set(HDF5_DEPS
    crypto
    curl
    pthread
    sz
    z
    dl
    m
)

# Combine into a single variable you can link later
set(HYPERION_EXTRA_LIBS
    ${HDF5_STATIC}
    ${HDF5_DEPS}
)
