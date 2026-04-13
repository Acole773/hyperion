#!/bin/bash
# =========================
# Environment
# =========================
module purge                           # <-- ALWAYS purge first on Frontier

module load PrgEnv-amd                 # AMD compiler environment
module load amd/7.2.0
module load rocm/7.2.0                 # HIP/ROCm
module load craype-accel-amd-gfx90a    # Correct GPU target
module load hdf5/1.14.5-mpi            # HDF5 if needed

# Run directory
# =========================
BASE=$(pwd)/..
RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$BASE/results/frontier/hip/$RUN_ID

mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

# =========================
# Diagnostics
# =========================
echo "Running on host $(hostname)"
echo "Current working directory: $(pwd)"
echo "Job started at: $(date)"

# =========================
# Run
# =========================
cp $BASE/build-frontier-hip/src/hyperion .
export HYPERION_DATA_DIR=$BASE

rocprofv3 --runtime-trace -S -f csv -- ./hyperion

echo "Job finished at: $(date)"

