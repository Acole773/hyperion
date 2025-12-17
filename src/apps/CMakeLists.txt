add_executable(hyperion_gpu
    main_gpu.c
)

target_link_libraries(hyperion_gpu
    PRIVATE
        hyperion_core
        burner_cpu
        burner_gpu
)

target_include_directories(hyperion_gpu
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

