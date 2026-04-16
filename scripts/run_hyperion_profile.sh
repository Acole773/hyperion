#!/bin/bash
#SBATCH -A ast218                     # Project allocation
#SBATCH --reservation=hackathon2
#SBATCH -J hyperion_parallel_150      # Job name
#SBATCH -o %x-%j.out                  # Output file
#SBATCH -t 00:30:00                   # Walltime
#SBATCH -p batch                      # Batch queue
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One MPI task
#SBATCH -c 1                          # One CPU core
#SBATCH --gres=gpu:1                  # <-- You MUST request a GPU


# =========================
# Environment
# =========================
#module reset 

#module load cpe/26.03
#module load PrgEnv-amd
#module load amd/7.2.0
#module load rocm/7.2.0
#module load craype-accel-amd-gfx90a
#module load cray-hdf5-parallel

#module unload darshan-runtime
#module unload cray-libsci/26.03.0

#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

source env_setup.txt

ml -t

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
#rocprofv3 --runtime-trace -- ./hyperion
rocprofv3 --pmc VALUUtilization OccupancyPercent -f csv --kernel-trace --stats -- ./hyperion 440


echo "Job finished at: $(date)"

