#!/bin/bash
#SBATCH -A ast218                     # Project allocation
#SBATCH -J sweep		      # Job name
#SBATCH -o sweep_%A_%a.out            # Output file
#SBATCH -t 00:30:00                   # Walltime
#SBATCH -p batch                      # Batch queue
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One MPI task
#SBATCH -c 1                          # One CPU core
#SBATCH --gpus=1 	              # <-- You MUST request a GPU
#SBATCH --array=0-25		      # Number of parallel JOBS

params=(110 120 210 220 230 320 330 340 430 440 450 540 550 560 650 660 670 760 770 780 870 880 890 980 990 1000)

# =========================
# Environment
# =========================
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

srun $EXEC $p > result_${p}.txt 2>&1

echo "Finished param $p"
