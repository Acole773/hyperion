add_library(burner_cpu
    burner_cpu.c
)

target_include_directories(burner_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

find_library(MATH_LIB m)

target_link_libraries(burner_cpu
    PRIVATE hyperion_core ${MATH_LIB}
)

target_compile_features(burner_cpu PUBLIC c_std_11)

