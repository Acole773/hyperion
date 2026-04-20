# Tell CMake this is HIP code
set_source_files_properties(
    bn_burner_gpu.c
    PROPERTIES LANGUAGE HIP
)

find_package(hip REQUIRED)

add_library(burner_gpu STATIC
    bn_burner_gpu.hip # HIP kernel
    bn_burner_gpu.c   # wrapper / launcher
    hip_debug_kernel.hip # <--- kernel definitions
    hip_wrapper.cpp
    hip_backend.c   # <-- holds kill command
)

target_include_directories(burner_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(burner_gpu
    PUBLIC
        hyperion_core
        hip::host
)
# HIP needs C++
set_target_properties(burner_gpu PROPERTIES
    HIP_STANDARD 14
    CXX_STANDARD 14
)

target_compile_features(burner_gpu PUBLIC c_std_11)

# Emit gfx assembly for the HIP source files so we can inspect the generated
# code after every build. --save-temps=obj writes temps next to the object
# file, avoiding name collisions between bn_burner_gpu.c and bn_burner_gpu.hip.
# NOTE: We apply this via source-file properties (compile-only) to avoid
# forwarding --save-temps to the AMDGPU device linker (lld), which has a
# known bus-error crash when it receives this flag in rocm/7.2.0.
set_property(SOURCE bn_burner_gpu.hip APPEND_STRING PROPERTY COMPILE_FLAGS
    " --save-temps=obj -g ")
set_property(SOURCE hip_debug_kernel.hip APPEND_STRING PROPERTY COMPILE_FLAGS
    " --save-temps=obj -g ")
