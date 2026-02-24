cmake_minimum_required(VERSION 3.21)

project(hyperion LANGUAGES C CXX HIP)

option(ENABLE_GPU "Enable GPU build" ON)
set(USE_GPU "HIP" CACHE STRING "GPU backend (HIP or CUDA)")

# Define reaction network size
if (NOT DEFINED SIZE)
	set(SIZE 16 CACHE STRING "Reaction network size")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
# Pass debugging flags to the HIP compiler
# -g: generates host debug symbols
# -O0: disables optimization for both host and device
# -ggdb: adds extra GDB-specific information
set(CMAKE_HIP_FLAGS_DEBUG "-g -O0 -ggdb")

add_compile_definitions(SIZE=${SIZE})

# Add main source directory
add_subdirectory(src)
#add_subdirectory(tests)

