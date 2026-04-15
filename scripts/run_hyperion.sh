#!/bin/bash
#SBATCH -A ast218                     # Project allocation
#SBATCH -J hyperion_parallel_150      # Job name
#SBATCH -o %x-%j.out                  # Output file
#SBATCH -t 00:30:00                   # Walltime
#SBATCH -p batch                      # Batch queue
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One MPI task
#SBATCH -c 1                          # One CPU core
#SBATCH --gres=gpu:1                  # <-- You MUST request a GPU
#SBATCH --exclusive                   # Optional: full node for debugging


# =========================
# Environment
# =========================
#module purge                           # <-- ALWAYS purge first on Frontier

#module load PrgEnv-amd                 # AMD compiler environment
#module load rocm/6.2.4                 # HIP/ROCm
#module load craype-accel-amd-gfx90a    # Correct GPU target
#module load hdf5/1.14.5-mpi            # HDF5 if needed
#module load darshan-runtime/3.4.7-mpi  # Darshan handles I/O for Frontier

source env_setup.txt

# =========================
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
srun ./hyperion

echo "Job finished at: $(date)"

