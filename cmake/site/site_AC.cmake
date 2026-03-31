# HDF5 paths (your system)
set(HDF5_INCLUDE_DIR "/usr/include/hdf5/serial")
set(HDF5_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu/hdf5/serial")

# Static libs (since you're using .a)
set(HDF5_LIBRARIES
    ${HDF5_LIBRARY_DIR}/libhdf5_hl.a
    ${HDF5_LIBRARY_DIR}/libhdf5.a
    crypto curl pthread sz z dl m
)
#//NOTE in CMake, you must ensure:m (math library) is last in link order

# Apply globally (simple approach for now)
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIR})
