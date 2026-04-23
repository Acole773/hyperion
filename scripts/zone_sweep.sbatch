#!/bin/bash
# Zone-sweep driver: one array task per zone count.
#
# Usage:
#   sbatch scripts/zone_sweep.sbatch              # defaults (2 iterations)
#   ITERS=4 sbatch scripts/zone_sweep.sbatch      # override iteration count
#
# After the array completes, tabulate with:
#   scripts/tabulate_zone_sweep.sh <array_job_id>
#
#SBATCH -J hyp_zsweep
#SBATCH -o zone_sweep-%A_%a.out
#SBATCH -t 00:30:00
#SBATCH -p MI250
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

set -e

# Zone counts chosen to probe scaling across the 104-CU device:
#   0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10 zones/CU
ZONES=(52 104 156 208 260 312 416 520 780 1040)
ITERS=${ITERS:-2}

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"

module load rocm/7.2.0
module load HDF5/1.14.0-ompi-4.1.5

BASE=$(pwd)
export HYPERION_DATA_DIR=$BASE
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:${LD_LIBRARY_PATH}

z=${ZONES[$SLURM_ARRAY_TASK_ID]}
if [ -z "$z" ]; then
    echo "Invalid array index $SLURM_ARRAY_TASK_ID (ZONES has ${#ZONES[@]} entries)"
    exit 1
fi

echo "Sweep config: zones=${z} iters=${ITERS}"

$BASE/build-local/src/hyperion $z $ITERS
