add_subdirectory(parse-data)
add_subdirectory(core)
add_subdirectory(cpu)

if(ENABLE_GPU)
    add_subdirectory(hipcore)
endif()

# Example top-level executable for testing
add_executable(hyperion_test main_gpu.c)

if(ENABLE_GPU)
    target_link_libraries(hyperion_test
        PRIVATE burner_gpu
    )
else()
    target_link_libraries(hyperion_test
        PRIVATE burner_cpu
    )
endif()
