add_library(hyperion_core
    init.c
    kill.c
    read_values.c
    store.c
)
target_include_directories(hyperion_core
    PUBLIC ${PROJECT_SOURCE_DIR}/src
)
target_sources(hyperion_core PRIVATE paths.c)
target_compile_features(hyperion_core PUBLIC c_std_11)
target_compile_definitions(hyperion_core
    PUBLIC 
    	SIZE=${SIZE}
	HYPERION_DATA_DIR="${PROJECT_SOURCE_DIR}"
)
target_link_libraries(hyperion_core
	PUBLIC hyperion_parse
)
#target_compile_definitions(core PUBLIC SIZE=${SIZE})
