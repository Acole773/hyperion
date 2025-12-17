add_library(hyperion_core
    init.c
    kill.c
    store.c
)
target_include_directories(hyperion_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(hyperion_core PUBLIC c_std_11)
target_compile_definitions(hyperion_core
    PUBLIC SIZE=${SIZE}
)
target_link_libraries(hyperion_core
    PUBLIC
        hyperion_parse
)
#target_compile_definitions(core PUBLIC SIZE=${SIZE})

