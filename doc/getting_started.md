# Hyperion: Getting Started Guide

This document provides a **quick-start user manual** for building and running **Hyperion** on the supported HPC platform.  
It is intended for new and returning users who want a **minimal, example-driven workflow** with no hardware-specific configuration.

---

## Initial Setup for New Users

This section describes the **one-time setup steps** required for a new user.  
Once completed, all future builds and runs follow the same workflow.

---

### 1). Create a User Directory

From the project workspace, create a directory using your **initials** and move into it.

**Example user:** John Doe → `JD`

```bash
mkdir JD
cd JD
```

---

### 2). Clone the Hyperion Repository

Clone the Hyperion repository into your user directory.

git clone <span style="color:#1f77b4; font-weight:bold;">"REPOSITORY_URL"</span>

Replace <span style="color:#1f77b4; font-weight:bold;">"REPOSITORY_URL"</span> with the official Hyperion repository URL.

After cloning, your directory structure should resemble:

```
JD/
└── hyperion/
    ├── CMakeLists.txt
    ├── src/
    ├── scripts/
    └── ...

```
---

### 3). Configure Absolute Paths in scripts/

The **only location** where user-specific absolute paths are required is in the scripts/ directory.

Navigate to:

```bash
cd hyperion/scripts
```

Locate the run scripts (e.g. run_hyperion.sh, run_debug_hyp.sh) and edit the base path.

---

### 4). Set the Base Directory

For **John Doe** (JD), define the base directory as:

BASE=/ccs/proj/ast218/<span style="color:#d62728; font-weight:bold; font-size:1.1em;">JD</span>/hyperion

Replace <span style="color:#d62728; font-weight:bold; font-size:1.1em;">JD</span>
with your own initials.

---

### 5). Set the Data Output Directory

Later in the same script, ensure the data directory is set consistently:

export HYPERION_DATA_DIR=/ccs/proj/ast218/<span style="color:#d62728; font-weight:bold; font-size:1.1em;">JD</span>/hyperion

---

### 6). Verify Setup

Once these paths are set:

● Proceed to the build instructions below

● Build Hyperion as usual

● Submit jobs using the provided scripts

No further user-specific configuration is required.

---

## Quick Start: Standard GPU Build and Run

This section demonstrates a *typical production build and execution* of Hyperion.

---

### 1). Load Required Modules

```bash
module purge
module load PrgEnv-amd
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load hdf5/1.14.5-mpi
```

---

### 2). Configure and Build Hyperion

```bash
rm -rf build-frontier-hip
mkdir build-frontier-hip
cd build-frontier-hip

cmake ../../hyperion \
  -DENABLE_GPU=ON \
  -DSIZE=150

make -j
```

**Note:**

<span style="color:#1f77b4; font-weight:bold;">SIZE=150</span> is a **user-controlled build parameter** and may be adjusted as needed.

### 3). Submit the Job

```bash
cd ..
cd scripts/
sbatch ./run_hyperion.sh
```
---
### 4). Monitor Job Status

squeue -u <span style="color:#d62728; font-weight:bold; font-size:1.1em;">YOUR_USERNAME</span>

Replace
<span style="color:#d62728; font-weight:bold; font-size:1.1em;">YOUR_USERNAME</span>
with your system username.

---

## Debug Build and Run

Use the debug configuration when developing, validating, or troubleshooting Hyperion.

### 1). Load Required Modules

```bash
module purge
module load PrgEnv-amd
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load hdf5/1.14.5-mpi
```

---

### 2). Configure and Build (Debug Mode)

```bash
rm -rf build-frontier-hip
mkdir build-frontier-hip
cd build-frontier-hip

cmake ../../hyperion \
  -DENABLE_GPU=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSIZE=150

make -j
```

Debug Notes:

● Enables debug symbols

● Disables aggressive compiler optimizations

● Intended for correctness and diagnostics, not performance

### 3). Submit the Debug Job

```bash
cd ..
cd scripts/
sbatch ./run_debug_hyp.sh
```

---

### 4). Monitor Job Status

squeue -u <span style="color:#d62728; font-weight:bold; font-size:1.1em;">YOUR_USERNAME</span>

---
## Summary

● One-time setup is limited to creating a user directory and editing scripts

● Absolute paths are required only in scripts/

● Standard and debug workflows are clearly separated

● All users follow the same build and execution process after setup

