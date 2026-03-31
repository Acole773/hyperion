cmake_minimum_required(VERSION 3.21)

project(hyperion LANGUAGES C CXX)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(ENABLE_GPU "Enable GPU (HIP) support" OFF)
set(SITE "default" CACHE STRING "Build configuration site")

# ------------------------------------------------------------
# Site configuration
# ------------------------------------------------------------
if(NOT ENABLE_GPU)
    if(NOT SITE)
        set(SITE "frontier") #change default choice here
    endif()
    message(STATUS "Using site configuration: ${SITE}")
    include(${CMAKE_SOURCE_DIR}/cmake/site/site_${SITE}.cmake OPTIONAL)
else()
    message(STATUS "GPU build: skipping site file (HIP compiler handles everything)")
endif()

# ------------------------------------------------------------
# GPU enable
# ------------------------------------------------------------
if(ENABLE_GPU)
    enable_language(HIP)
    message(STATUS "HIP enabled")
else()
    message(STATUS "CPU-only build")
endif()

# Define reaction network size
if (NOT DEFINED SIZE)
	set(SIZE 16 CACHE STRING "Reaction network size")
endif()

# ------------------------------------------------------------
# Default build type
# ------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified. Defaulting to RelWithDebInfo.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()


# ------------------------------------------------------------
# Debug flags
# ------------------------------------------------------------
if(ENABLE_GPU)

    enable_language(HIP)
    message(STATUS "HIP enabled, CMake will use hipcc for .hip and .cpp files")
	
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")

	message(STATUS "Debug build enabled")
	# -g: generates host debug symbols
	# -O0: disables optimization for both host and device
	# -ggdb: adds extra GDB-specific information
	set(CMAKE_HIP_FLAGS_DEBUG
    	    "-g -O0 -ggdb -fno-inline -fno-omit-frame-pointer"
	)

    else()

	message(STATUS "Optimized build enabled")

	set(CMAKE_HIP_FLAGS_RELEASE
            "-O3 -fno-strict-aliasing"
	)

    endif()
else()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Debug build enabled (CPU)")
	# need set target for cpu
    else()
        message(STATUS "Optimized build enabled (CPU)")
	# need set target for cpu
    endif()

endif()

# ------------------------------------------------------------
# Rest of your config...
# ------------------------------------------------------------

add_compile_definitions(SIZE=${SIZE})

# Add main source directory
add_subdirectory(src)
#add_subdirectory(tests)

