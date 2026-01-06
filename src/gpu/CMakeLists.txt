add_library(gpu_backend
    gpu_backend.c
)

target_include_directories(gpu_backend
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(gpu_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gpu_backend
    PUBLIC burner_gpu   # HIP or CUDA implementation
)

# Backend selection
if(ENABLE_GPU)
    target_compile_definitions(gpu_backend PUBLIC ENABLE_GPU)

    if(USE_GPU STREQUAL "HIP")
        target_compile_definitions(gpu_backend PUBLIC USE_HIP)
        target_link_libraries(gpu_backend PUBLIC burner_gpu)

    elseif(USE_GPU STREQUAL "CUDA")
        target_compile_definitions(gpu_backend PUBLIC USE_CUDA)
        target_link_libraries(gpu_backend PUBLIC burner_cuda)

    endif()

else()
    # CPU-only
    target_link_libraries(gpu_backend PUBLIC burner_cpu)
endif()

target_compile_features(gpu_backend PUBLIC c_std_11)

