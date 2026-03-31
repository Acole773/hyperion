# site_frontier.cmake — robust GPU/CPU setup for Frontier

# -----------------------------
# Default CPU compiler (only for CPU builds)
# -----------------------------
if(NOT DEFINED ENABLE_GPU OR NOT ENABLE_GPU)
    set(CMAKE_C_COMPILER "cc")
    set(CMAKE_CXX_COMPILER "CC")
endif()

# -----------------------------
# HDF5 (common)
# -----------------------------
set(HDF5_DIR "/sw/frontier/spack-envs/cpe24.11-cpu/opt/cce-18.0.1/hdf5-1.14.5-fbhfdr23kwzgknnufevewoslxgxfbvo7")
include_directories(${HDF5_DIR}/include)
link_directories(${HDF5_DIR}/lib)

# -----------------------------
# GPU / HIP
# -----------------------------
if(DEFINED ENABLE_GPU AND ENABLE_GPU)
    # Require ROCm module to be loaded externally
    find_program(HIPCC hipcc)
    if(NOT HIPCC)
        message(FATAL_ERROR "hipcc not found! Make sure you loaded ROCm module.")
    endif()

    # Add HIP includes if needed
    include_directories(
        ${CMAKE_SOURCE_DIR}/src/hipcore
        ${CMAKE_SOURCE_DIR}/src/core
        $ENV{ROCM_PATH}/hip/include
    )
    link_directories($ENV{ROCM_PATH}/hip/lib)

    message(STATUS "GPU build: using hipcc from environment ($HIPCC)")
endif()
