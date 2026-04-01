# Hyperion Agent Notes

## Overview

Hyperion is a thermonuclear reaction-network solver meant to be embedded into larger simulation codes such as Flash-X, Chimera, or the Einstein Toolkit. The core design is:

- parse text network/rate tables into global arrays
- precompute flux bookkeeping (`F+`, `F-`, reaction masks)
- evolve one or more zones with either the CPU burner or the HIP burner

`SIZE` is a compile-time switch. It selects both the network files and the hardcoded reaction-count constants:

- `SIZE=16`: alpha network, 48 reactions
- `SIZE=150`: 150-species network, 1604 reactions
- `SIZE=365`: 365-species network, 4395 reactions

For the hackathon case, `-DSIZE=150` is the important build, and the existing drivers use:

- `temp = 5e09`
- `dens = 1e08`
- `tstep = 1e-06`

Those values are hardcoded in both [`src/main.c`](/Users/hrh/hyperion_fork/src/main.c) and [`src/main_gpu.c`](/Users/hrh/hyperion_fork/src/main_gpu.c).

## Repo Map

- [`src/core/init.c`](/Users/hrh/hyperion_fork/src/core/init.c): chooses the network/rate files from `SIZE`, loads them, then calls `data_init()`.
- [`src/parse-data/parser.c`](/Users/hrh/hyperion_fork/src/parse-data/parser.c): parses `data/network-*.txt` and `data/ratelibrary-*.txt`, then builds the reaction-mask and `F+`/`F-` lookup structures.
- [`src/core/store.h`](/Users/hrh/hyperion_fork/src/core/store.h): global arrays and metadata shared across CPU and GPU paths.
- [`src/core/bn_burner.c`](/Users/hrh/hyperion_fork/src/core/bn_burner.c): CPU burner implementation.
- [`src/hipcore/bn_burner_gpu.c`](/Users/hrh/hyperion_fork/src/hipcore/bn_burner_gpu.c): HIP-side burner wrapper and device memory setup.
- [`src/gpu/gpu_backend.c`](/Users/hrh/hyperion_fork/src/gpu/gpu_backend.c): thin GPU backend layer used by the GPU executable.
- [`data/`](/Users/hrh/hyperion_fork/data): shipped reaction-network and rate-library inputs.
- [`scripts/run_hyperion.sh`](/Users/hrh/hyperion_fork/scripts/run_hyperion.sh): Frontier batch script example.
- `origin/CMAKE_rebuild:doc/getting_started.md`: the maintainer’s clearest statement of the intended future workflow.

## Data Path Contract

`HYPERION_DATA_DIR` should point to the repo root, not the `data/` directory itself. Runtime code appends `data/network-*.txt` and `data/ratelibrary-*.txt` internally.

Good:

```bash
export HYPERION_DATA_DIR=/path/to/hyperion
```

Bad:

```bash
export HYPERION_DATA_DIR=/path/to/hyperion/data
```

## Build Systems

### Legacy make

The repo still carries the original make-based build. On non-Frontier machines, use `SITE=default`.

Verified on 2026-04-01 on this `arm64` macOS host with Homebrew GCC 15:

```bash
make SITE=default CC=/opt/homebrew/bin/gcc-15 LD=/opt/homebrew/bin/gcc-15 serial_16
HYPERION_DATA_DIR=$PWD ./object/serial_16
```

If you want to avoid writing into the repo, override `OBJECT_DIR`, for example:

```bash
make SITE=default CC=/opt/homebrew/bin/gcc-15 LD=/opt/homebrew/bin/gcc-15 OBJECT_DIR=/tmp/hyperion_object serial_150
```

Status:

- `serial_16` built and ran successfully on this machine.
- `serial_150` built successfully on this machine.

### CMake

The maintainer said `origin/CMAKE_rebuild` is the branch to study for the future GPU ON/OFF build system, and `origin/CMAKE_rebuild:doc/getting_started.md` is the best available build guide.

For current local CPU work, this command path is verified on this machine:

```bash
cmake -S . -B /tmp/hyperion_cpu150 \
  -DENABLE_GPU=OFF \
  -DSIZE=150 \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15
cmake --build /tmp/hyperion_cpu150 -j4
/tmp/hyperion_cpu150/src/hyperion
```

Status:

- `SIZE=16` built and ran successfully.
- `SIZE=150` built and ran successfully.
- The `SIZE=150` run completed on this host, but it took noticeably longer than `SIZE=16`.

## Verified Local Findings

- This machine has no `hipcc`, so GPU/HIP builds were not testable here.
- Apple Clang is what `gcc` resolves to here; use `/opt/homebrew/bin/gcc-15` and `/opt/homebrew/bin/g++-15` for GCC builds.
- The CPU code is now usable on this ARM Mac, but older assumptions in the repo were very x86/HIP-centric.

## Known Gaps

- The Python wrapper/test path in [`hypy/`](/Users/hrh/hyperion_fork/hypy) was not runnable locally because the system Python does not have `Cython` installed.
- [`tests/test.py`](/Users/hrh/hyperion_fork/tests/test.py) and [`tests/testmod_physical.py`](/Users/hrh/hyperion_fork/tests/testmod_physical.py) look intended for the 150-species CPU burner, but the wrapper/tooling around them is still brittle.
- [`src/core/diagnostic.h`](/Users/hrh/hyperion_fork/src/core/diagnostic.h) intentionally redefines `__DIAG_HALT`, which produces build warnings in `bn_burner.c`.
- [`src/apps/CMakeLists.txt`](/Users/hrh/hyperion_fork/src/apps/CMakeLists.txt) appears stale relative to the top-level CMake flow and was not part of the verified build path.
- Some job scripts and older logs still refer to `hyperion_test`; check [`src/CMakeLists.txt`](/Users/hrh/hyperion_fork/src/CMakeLists.txt) for the current executable name before copying commands between branches.
- Frontier scripts in [`scripts/`](/Users/hrh/hyperion_fork/scripts) are cluster-specific and hardcode modules, allocation, and filesystem paths.

## Practical Advice

- For CPU debugging or algorithm work on a non-Frontier machine, start with CMake and `-DENABLE_GPU=OFF -DSIZE=150`.
- For Frontier GPU work, compare any CMake edits against `origin/CMAKE_rebuild` first; that branch is clearly where the maintainer has been concentrating build-system changes.
- If you revive the Python path, install `Cython` first and then normalize the `libhyburn` target/library assumptions before trusting `make test`.
