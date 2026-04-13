#!/bin/bash
module reset

module load cpe/26.03
module load PrgEnv-amd
module load amd/7.2.0
module load rocm/7.2.0
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel

module unload darshan-runtime
module unload cray-libsci/26.03.0

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

ml -t

cmake -S . -B build-frontier-hip -DENABLE_GPU=ON -DSIZE=150
cmake --build build-frontier-hip -j8
