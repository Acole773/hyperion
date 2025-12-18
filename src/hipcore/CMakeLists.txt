add_library(burner_gpu
    bn_burner_gpu.c
)

# Tell CMake this is HIP code
set_source_files_properties(
    bn_burner_gpu.c
    PROPERTIES LANGUAGE HIP
)

target_include_directories(burner_gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core
)

target_link_libraries(burner_gpu
    PUBLIC hyperion_core
)

target_compile_features(burner_gpu PUBLIC c_std_11)

