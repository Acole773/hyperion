# Tell CMake this is HIP code
set_source_files_properties(
    bn_burner_gpu.c
    PROPERTIES LANGUAGE HIP
)

add_library(burner_gpu STATIC
    bn_burner_gpu.hip # HIP kernel
    bn_burner_gpu.c   # wrapper / launcher
    hip.c
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
    PUBLIC hyperion_core
)
# HIP needs C++
set_target_properties(burner_gpu PROPERTIES
    HIP_STANDARD 14
    CXX_STANDARD 14
)

target_compile_features(burner_gpu PUBLIC c_std_11)

