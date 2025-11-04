#!/bin/bash
#SBATCH -A AST218             # <-- Replace with your project allocation
#SBATCH -J hyperion_serial150  # Job name
#SBATCH -o %x-%j.out          # Output file (%x = job name, %j = job ID)
#SBATCH -t 00:30:00           # Walltime (hh:mm:ss)
#SBATCH -p batch              # Partition/queue to submit to
#SBATCH -N 1                  # Number of nodes
#SBATCH -n 1                  # Number of MPI ranks (serial run)
#SBATCH -c 1                  # Number of CPUs per task
#SBATCH --exclusive           # Optional: request full node for clean testing

# Load your environment (compiler, MPI, HDF5)
module load PrgEnv-cray
module load hdf5/1.14.5-mpi

# Navigate to your working directory
cd $SLURM_SUBMIT_DIR

# (Optional) print some diagnostic info
echo "Running on host $(hostname)"
echo "Current working directory: $(pwd)"
echo "Job started at: $(date)"

# Run the executable
srun ./object/serial_150

echo "Job finished at: $(date)"

