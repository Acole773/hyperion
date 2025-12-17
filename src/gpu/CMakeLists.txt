# gpu target library (HIP or CUDA)
if(USE_GPU STREQUAL "HIP")
  # compiles .hip with HIP
  enable_language(HIP)

	add_library(burner_gpu
    	bn_burner_gpu.hip
    	bn_burner_gpu.c   # wrapper / launcher
	)

	target_include_directories(burner_gpu
    	PUBLIC
        	${CMAKE_CURRENT_SOURCE_DIR}
        	${PROJECT_SOURCE_DIR}/core
	)

	target_link_libraries(burner_gpu
    		PRIVATE hyperion_core
	)

	# HIP needs C++
	set_target_properties(burner_gpu PROPERTIES
    		HIP_STANDARD 14
    		CXX_STANDARD 14
	)

  # pass HIP-specific compile options if needed
elseif(USE_GPU STREQUAL "CUDA")
  add_library(burner_gpu STATIC hip_wrapper.cpp bn_burner_gpu.cu)
  target_compile_definitions(burner_gpu PUBLIC SIZE=${SIZE})
  target_include_directories(burner_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(burner_gpu PUBLIC core)
else()
  add_library(burner_gpu INTERFACE) # no GPU
  target_link_libraries(burner_gpu INTERFACE burner_cpu)
endif()
target_compile_definitions(hyperion_core
    PUBLIC SIZE=${SIZE}
)


