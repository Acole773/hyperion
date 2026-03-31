# ------------------------------------------------------------
# Local CPU/GPU build defaults (CMake-friendly)
# ------------------------------------------------------------

# CPU compiler
set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -std=c99")

# Optional SIMD
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -march=native -D__HYPERION_USE_SIMD")

# Linker flags (library libhyburn.a must be in OBJECT_DIR)
# target_link_libraries(hyperion PRIVATE ${OBJECT_DIR}/libhyburn.a m)

# GPU compiler (HIP)
# set(CMAKE_C_COMPILER hipcc)  # only enable if building GPU target
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ...")
