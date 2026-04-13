#!/bin/bash
#SBATCH -A ast218                     # Project allocation
#SBATCH -reservation = hackathon1     # Hackathon reservation 
#SBATCH -J sweep		      # Job name
#SBATCH -o sweep_%A_%a.out            # Output file
#SBATCH -t 00:30:00                   # Walltime
#SBATCH -p batch                      # Batch queue
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One MPI task
#SBATCH -c 1                          # One CPU core
#SBATCH --gpus=1 	              # <-- You MUST request a GPU
#SBATCH --array=0-4		      # Number of parallel JOBS

params=(64 128 256 512 1024)

# =========================
# Environment
# =========================
module purge                           # <-- ALWAYS purge first on Frontier

module load PrgEnv-amd                 # AMD compiler environment
module load rocm/6.2.4                 # HIP/ROCm
module load craype-accel-amd-gfx90a    # Correct GPU target
module load hdf5/1.14.5-mpi            # HDF5 if needed
#module load darshan-runtime/3.4.7-mpi  # Darshan handles I/O for Frontier

# =========================
# Paths
# =========================
BASE=$(pwd)/..
RESULTS_DIR=$BASE/results/frontier/hip/${SLURM_ARRAY_JOB_ID}
EXEC=$BASE/build-frontier-hip/src/hyperion

mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

# =========================
# Diagnostics
# =========================
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host $(hostname)"
echo "Current working directory: $(pwd)"

# Run
# =========================
export HYPERION_DATA_DIR=$BASE

p=${params[$SLURM_ARRAY_TASK_ID]}

if [ -z "$p" ]; then
    echo "Invalid array index"
    exit 1
fi

rocprofv3 --kernel-trace --stats -- $EXEC $p > result_${p}.txt 2>&1

echo "Finished param $p"
