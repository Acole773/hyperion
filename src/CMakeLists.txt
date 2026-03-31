# ------------------------------------------------------------
# Core components (always needed)
# ------------------------------------------------------------
add_subdirectory(parse-data)
add_subdirectory(core)

# ------------------------------------------------------------
# GPU components (only if enabled)
# ------------------------------------------------------------
if(ENABLE_GPU)
	#add_subdirectory(gpu) # <-- this defines burner_gpu
	add_subdirectory(hipcore)
	add_subdirectory(gpu)
endif()

# ------------------------------------------------------------
# Select executable based on GPU flag
# ------------------------------------------------------------
if(ENABLE_GPU)

    message(STATUS "Building GPU executable")

    add_executable(hyperion main_gpu.c)

    add_dependencies(hyperion gpu_backend)

    target_link_libraries(hyperion
        PRIVATE
            gpu_backend
            hyperion_core
            hyperion_parse
    )

else()

    message(STATUS "Building CPU executable")

    add_executable(hyperion
	main.c
	core/init.c
	core/kill.c
	core/store.c
	core/bn_burner.c
    )

    target_link_libraries(hyperion
        PRIVATE
            hyperion_core
            hyperion_parse
	    m
    )

endif()
