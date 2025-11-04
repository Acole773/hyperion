#!/bin/bash
#SBATCH -A ast218        # Replace with your project/allocation
#SBATCH -J hello_gpu                # Job name
#SBATCH -o hello_gpu.out            # Standard output
#SBATCH -e hello_gpu.err            # Standard error
#SBATCH -p batch                    # Partition
#SBATCH -t 00:10:00                 # Max walltime
#SBATCH -N 1                         # Number of nodes
#SBATCH --gpus-per-node=8           # GPUs per node (Frontier has 8 per node)

# Load your modules
module load PrgEnv-amd/8.6.0
module load rocm/6.2.4

# Run the executable
srun ./hello_gpu

