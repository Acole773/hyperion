cmake_minimum_required(VERSION 3.21)

if (NOT DEFINED SIZE)
    message(FATAL_ERROR "SIZE must be defined (e.g. -DSIZE=16)")
endif()

add_compile_definitions(SIZE=${SIZE})

project(hyperion LANGUAGES C CXX HIP)

option(ENABLE_GPU "Enable GPU build" ON)
set(SIZE 16 CACHE STRING "Reaction network size")

add_subdirectory(src)

