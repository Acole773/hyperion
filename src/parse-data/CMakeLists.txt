add_library(hyperion_parse
    parser.c
)

target_include_directories(hyperion_parse
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(hyperion_parse PUBLIC c_std_11)

