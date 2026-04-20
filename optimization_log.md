# `hyperion_burner_dev_kernel` Optimization Log

Iterative optimization of the `hyperion_burner_dev_kernel` on AMD MI250 (gfx90a).
Stop condition: **10 unsuccessful attempts in a row** (explore a wide range of optimization types).

## Ground rules

- Precision: stay in `double`; allow safe fp64 reassociation flags but no fp32 intrinsics.
- Correctness check: `xout[]` and `sdotrate[]` must match the baseline within **1e-12 relative tolerance**.
- Build: `sbatch build.sh` (gfx90a, `-O3 -g --save-temps=obj -fno-strict-aliasing`); artifacts in `build-local/`.
- Timing run: `sbatch run.sh <zones> <iterations>` — 1 iteration for quick comparisons, 14 for profiling.
- Profiling:
  - `sbatch compute_profile.sh 150 14` → `rocprof-compute profile -k hyperion_burner_dev_kernel --iteration-multiplexing --device 0`.
  - Assembly: `build-local/src/hipcore/CMakeFiles/burner_gpu.dir/bn_burner_gpu-hip-amdgcn-amd-amdhsa-gfx90a.s` (inspect VGPR/SGPR, scratch spills, `s_waitcnt`, `v_cndmask`, indirect loads, `__ocml_*` expansions).
- Each optimization:
  1. Snapshot the baseline.
  2. Apply one focused change.
  3. Rebuild.
  4. Single-iter run → compare avg GPU time.
  5. If faster: commit on `kernel-optimization-experiments`, keep; otherwise revert.
  6. Reprofile with `rocprof-compute` only when a change is kept (or when an unexpected result needs explanation).
  7. Record observation below.

## Baseline metrics (for reference)

Source of reference (from prior runs with 14 iterations, zones=8 — TO BE REPLACED with a fresh 150-zone baseline):

| Metric                         | Value (prior 14-iter/8-zone run)   |
| ------------------------------ | ---------------------------------- |
| Avg wall time per iter         | 45.90 s                            |
| Avg GPU time per iter          | 45.90 s                            |
| CU utilization                 | 7.6%                               |
| VALU active threads            | 21.6%                              |
| Occupancy                      | 4 waves/CU (of 8 possible)         |
| Memory coalescing              | 35.8%                              |
| LDS bank conflicts / access    | 0.14                               |

> The 8-zone measurement was taken with `BATCHCNT`/default zones = 8. The optimization experiments will use **104 zones** as the default workload and 1 iteration for quick timing; 14 iterations only when metrics/PMCs are needed. (440 zones was tried but the single-iteration run was taking too long for an interactive-pace experiment loop.)

## Experiments

### Experiment 00 — Baseline (fresh measurement)

**Intent**: Establish a fresh baseline timing on 104 zones with the current code; snapshot the kernel assembly and register usage.

**Change**: None (reference build).

**Code version**: branch `kernel-optimization-experiments` at start (tip = `b4771b1` + CLI-iterations + `--save-temps=obj` build flag).

**Timing (104 zones, 1 iter)**:

| avg_wall_time | avg_gpu_time |
| ------------: | -----------: |
|    46.701679 s |  46.701652 s |

Output preserved in `baseline_output.txt`. Assembly preserved in `asm_baseline/baseline.s`.

**Kernel metadata (from `.s`)**:

| Metric                    | Value  |
| ------------------------- | -----: |
| VGPRs (`NumVgprs`)        |   117  |
| SGPRs (`TotalNumSgprs`)   |   100  |
| Scratch (`ScratchSize`)   |     0  |
| Occupancy (waves/EU)      |     4  |
| Code size                 | 16,056 bytes |
| LDS (static)              |  8 B/WG + dyn |

**Assembly notes**:

- **`pow(tmp, 1.0/3.0)` at src line 107** is fully inlined from `__clang_hip_math.h:1025` and spans asm lines 621-956 (~335 lines, ~200+ instructions). Uses `v_frexp_mant_f64`, `v_ldexp_f64`, multiple `v_rcp_f64`, `v_fma_f64`, and several `v_cmp_*` to handle negative/special cases. This is by far the largest single expression in the kernel prologue and a top candidate for register-pressure reduction.
- **`exp(...)` at src line 131** (inside the rate loop) is fully inlined from `__clang_hip_math.h:831` and spans roughly asm lines 1213-1250 per rate loop iteration (~40 lines). Called once per reaction per timestep → dominant VALU cost.
- **No function calls** (`s_swappc_b64` count = 0) — `pow`, `exp`, `log` are all inlined.
- **No scratch spills** (`ScratchSize = 0`) — register allocation succeeded without spilling.
- **Occupancy = 4** limited by VGPR count (117 > 64-VGPR threshold for 8 waves/EU on gfx90a). Getting below 96 → 5 waves, below 80 → 6 waves, below 72 → 7 waves, ≤64 → 8 waves. Shaving 20-30 VGPRs would meaningfully improve occupancy.
- **Kernel prologue dominated by thermodynamic precompute** (`pow`, `log`, division/reciprocals, `exp` setup), executed once per zone; the inner `while` loop re-reads `rate[r]` from global memory each sweep but `rate[]` itself is only computed once.

**rocprof-compute**: Skipped for baseline; previous runs already characterised CU util ≈ 7.6%, VALU active ≈ 21.6%, occupancy 4/8, coalescing ≈ 35.8% (at 8 zones — will re-measure if needed at 104 zones).

**Decision**: N/A (reference).

---

### Experiment 01 — `pow(tmp, 1/3)` → `cbrt(tmp)`

**Hypothesis**: `pow(x, y)` on AMDGPU double expands to `exp(y*log(x))` plus special-case handling (sign, NaN, integer-y optimization). `cbrt()` has a dedicated ocml intrinsic with shorter instruction sequence and fewer live-range-extending constants, reducing register pressure around line 107 of `bn_burner_gpu.hip`.

**Change**: `double t93 = pow(tmp, THIRD);` → `double t93 = cbrt(tmp);`

**Timing (104 zones, 1 iter)**:

| metric         |   baseline |   exp01 |    Δ |
| -------------- | ---------: | ------: | ---: |
| avg_gpu_time   |   46.7017 s | 46.7434 s | +0.09 % |

**Assembly notes**:

| Metric                    | baseline |  exp01 |   Δ |
| ------------------------- | -------: | -----: | --: |
| VGPRs                     |      117 |    103 | −14 |
| SGPRs                     |      100 |    100 |   0 |
| ScratchSize               |        0 |      0 |   0 |
| Occupancy (waves/EU)      |        4 |      4 |   0 |
| Code size (bytes)         |  16,056  | 14,736 | −1,320 (−8.2%) |

The `pow` inline block (~335 asm lines in baseline) is replaced by a much shorter `cbrt` sequence. No jump in occupancy because VGPR count still exceeds the 96-VGPR ceiling for 5 waves/EU.

**Correctness**: `xout[]` and `sdotrate[]` **bit-identical** to baseline (diff is empty apart from the timing line).

**Why no timing improvement**:
- `pow(tmp, 1/3)` is executed **once per zone** in the prologue, not in the hot inner `while` loop. The ~200 instructions saved are amortised across the entire zone computation and don't move the needle.
- Occupancy didn't change (still 4 waves/EU) — VGPR ceiling wasn't crossed.

**Decision**: **Keep** the change anyway — strictly cleaner code, lower VGPR headroom for future optimizations, bit-identical correctness. **But counted as 1 unsuccessful experiment for the "yields better performance" criterion** (1/5).

---

### Experiment 02 — Rewrite `exp(x)` as `exp2(x * log2(e))`

**Hypothesis**: `exp(double)` lowers through `__ocml_exp_f64` which internally does `exp2(x * log2(e))` with range reduction. By expressing the computation as an explicit `exp2()` call, we give the compiler a fused-multiply-add opportunity and skip redundant range-reduction constants. May reduce SGPRs (fewer constant tables) and VGPRs.

**Change**: in rate loop line 129-137, replace `exp(...)` with `exp2((...) * 1.4426950408889634)`.

**Timing**: _TBD_

**Assembly notes**: _TBD_

**Decision**: _TBD_

---

### Experiment 03 — Add `__restrict__` to kernel pointer params

**Hypothesis**: Without `__restrict__`, the compiler must assume aliasing between `xout`, `xin`, `flux`, `rate_g`, etc. This forces reloads and blocks LICM/CSE. Marking all 24 pointer params restrict should reduce redundant loads and free registers.

**Change**: `__restrict__` added to every pointer parameter of `hyperion_burner_dev_kernel`.

**Assembly metrics** (vs exp01 baseline):

| Metric     | exp01 |  exp03 |  Δ |
| ---------- | ----: | -----: | -: |
| VGPRs      |   103 |    103 |  0 |
| SGPRs      |   100 |    100 |  0 |
| Occupancy  |     4 |      4 |  0 |

Register allocation didn't change, suggesting the compiler was already doing adequate alias analysis.

**Timing (3-run avg, multiple nodes)**:

| config                 | C55       | C56       | C59       | avg       |
| ---------------------- | --------: | --------: | --------: | --------: |
| baseline re-run        | 46.725    | 46.738    | 46.575    | **46.679**|
| exp03 `__restrict__`   | 49.139 / 49.177 | 49.193 | —         | **49.17** |

Δ = **+5.3% slower**, reproducible across nodes.

**Correctness**: bit-identical.

**Why slower**: the compiler, freed from aliasing constraints, most likely re-scheduled memory operations — possibly issuing more loads upfront (hoisting via LICM) and degrading latency-hiding interleave between loads and arithmetic. Another plausible factor: without aliasing fences, more speculative reads could be issued earlier, increasing outstanding-memory-op queue depth and contending with the indirect gathers in the hot loops. Same lesson as exp02: this kernel is already near a memory/cache bandwidth sweet spot, and "helpful" compiler transformations can push it over the cliff.

**Decision**: **Reverted** — counted as unsuccessful **3/10**.

---

### Experiment 02 — Remove `isfinite(f_rate)` printf guard from hot `while` loop

**Hypothesis**: The `isfinite(f_rate)` check + `printf` branch at line 160-162 adds a divergent branch and keeps `printf` args (incl. `blockIdx.x`, `r`, `f_rate`) live across the hot rate-compute path. Removing it should simplify the loop body and reduce register pressure.

**Change**: removed the isfinite check block entirely.

**Timing (104 zones, 1 iter, 3-run avg)**:

| metric         |   baseline |            exp02 |    Δ |
| -------------- | ---------: | ---------------: | ---: |
| avg_gpu_time   |  46.7017 s | **48.7429 s (+4.4 %)** — 3-run avg: {48.769, 48.783, 48.676} σ≈0.06 |

**Assembly notes**:

| Metric                    | exp01 (prev) |  exp02 |   Δ |
| ------------------------- | -----------: | -----: | --: |
| VGPRs                     |          103 |     93 | −10 |
| SGPRs                     |          100 |    100 |   0 |
| Occupancy (waves/EU)      |            4 |      5 |  +1 |

**Correctness**: `xout[]`/`sdotrate[]` bit-identical.

**Why slower despite better resource metrics** — this is the important finding:

- VGPRs dropped 10 and occupancy went up 4 → 5 waves/EU, yet **GPU time increased ~2 s per iteration, reproducibly**.
- Kernel metadata already flagged `MemoryBound: 0` — the compiler judged it VALU-bound. But the **observed behaviour contradicts that**: adding a 5th concurrent wave hurt performance.
- Most plausible explanation: the kernel is **memory-latency-bound on indirect gathers** — specifically `xout_zone[reactant_{1,2,3}[r]]` in the flux loop and `flux[f_plus_map[j]]` / `flux[f_minus_map[j]]` in the species-update loop. These are scatter-like reads that don't coalesce (baseline coalescing was 35.8%). With 4 waves/EU, each wave has more L1/L2 cache share and shorter queue depth; at 5 waves/EU, the additional wave competes for the same finite cache capacity, causing **more misses and more serialised memory traffic**.
- The observed "MemoryBound: 0" is a static compiler heuristic; the actual runtime behaviour is latency- and cache-limited.

**Decision**: **Reverted** — clean timing regression even though the resource metrics "look" better. Counted as **unsuccessful 2/10**.

**Key takeaway for subsequent experiments**: pushing occupancy up is NOT a good strategy here. Instead, focus on:
1. Reducing memory traffic (less redundant loads, cache-friendly access patterns).
2. Improving L1/L2 hit rates (reusing loaded data in registers).
3. Converting indirect accesses to linear / coalesced where possible.
4. **Lowering** occupancy-sensitive VGPR budget may even help (e.g. if we can stay at 4 waves/EU).

---

### Experiment 04 — Cache `num_react_species[r]` in a local

**Hypothesis**: Inside the flux loop, `num_react_species[r]` is read 3 times per reaction. Caching once to a register should cut global loads.

**Change**: `int nrs = num_react_species[r];` then use `nrs` in the three conditionals.

**Assembly metrics**: unchanged (VGPR 103, SGPR 100, occupancy 4, codeLen 14736). The compiler was already doing CSE on the three loads.

**Timing (3-run avg)**: 46.737 s (baseline 46.679 s, Δ +0.12 %, noise).

**Correctness**: bit-identical.

**Decision**: **Reverted** — no measurable gain since compiler already CSE'd. Counted as unsuccessful **4/10**.

---

### Experiment 05 — Block size 256 → 128

**Hypothesis**: With memory-latency-bound kernel, lower occupancy (more cache per wave) may help. Also each thread doing 2x work per iteration gives compiler more ILP opportunity.

**Change**: `blockdim(128,1,1)`; generalize the cross-wave reduction to use `num_waves = blockDim.x >> 6` instead of hard-coded 4.

**Assembly metrics**: VGPRs 104 (+1), occupancy 4 waves/EU (same).

**Timing (3-run avg)**: 50.563 s, Δ **+8.3 %** reproducibly.

**Correctness**: bit-identical.

**Why slower**: halving block size doubles per-thread work in the flux and species-update loops. Even though LDS budget per block halves (to ~6.4 KB), the doubled per-thread iteration count means more global loads per thread before they can overlap, and reduced warps per block hurts LDS bank conflict averaging. The run-time of the outer while-loop scales with per-thread iteration count, not with wave count.

**Decision**: **Reverted** — counted as unsuccessful **5/10**. (Keeping the generalized cross-wave reduction code as a minor cleanup, since it works identically for 256 threads.)

---

### Experiment 06 — **Promote `rate[]` to LDS** ✅ FIRST SUCCESS

**Hypothesis**: `rate[r]` is computed once per zone in the prologue but **re-read from global memory every timestep** inside the hot while-loop (NUM_REACTIONS × ~1000 timesteps per zone). Moving rate to LDS eliminates this repeated global traffic without increasing occupancy (LDS budget stays within 2 blocks/CU).

**Change**:

- Kernel: declared a new LDS pointer `rate = reduce_buf + num_waves_k` pointing into the existing dynamic shared-memory block; dropped the `rate_g + zone*NUM_REACTIONS` offset.
- Launcher: bumped `sharedmem_allocation` by `NUM_REACTIONS * sizeof(double)` (≈12.8 KB; total LDS per block: ~25.6 KB, still fits 2 blocks/CU with 64 KB LDS/CU).

**Assembly metrics**:

| Metric    | baseline (exp01) |  exp06 |   Δ |
| --------- | ---------------: | -----: | --: |
| VGPRs     |              103 |    105 |  +2 |
| SGPRs     |              100 |    100 |   0 |
| Occupancy |                4 |      4 |   0 |

Register count barely moved, occupancy preserved — the change is purely in where `rate[r]` lives.

**Timing (3-run avg across C55, C56, C59)**:

| metric         |   baseline |     exp06 |       Δ |
| -------------- | ---------: | --------: | ------: |
| avg_gpu_time   |  46.679 s  |  46.352 s | **−0.70 %** |

Per-node: C55 46.366, C56 46.380, C59 46.310 — consistent.

**Correctness**: `xout[]` and `sdotrate[]` **bit-identical** to baseline.

**Why it works**: The outer `while` loop inside each zone executes many time-steps (adaptive dt). For each time-step, every thread reads `rate[r]` once from global memory. LDS reads are ~10× lower latency than uncached global reads and don't compete with the indirect-gather global traffic on `xout_zone[reactant_X[r]]` and `flux[f_plus_map[j]]`. Even though the speedup is modest (~0.7 %), it's a real and reproducible win that reduces memory pressure on the exact bottleneck we identified from exp02/03.

**Decision**: **KEPT**. Failure counter reset from 5 → 0.

---

### Experiment 07 — Scoped fast-math for the HIP TU

**Hypothesis**: `-fno-math-errno -ffinite-math-only -fno-signed-zeros` lets the compiler drop errno handling and NaN/Inf branches inside `exp`/`log`/`cbrt`, trimming instructions.

**Change**: added the three flags to `bn_burner_gpu.hip` via `set_property COMPILE_FLAGS`.

**Assembly metrics**: VGPRs **105 → 89 (−16)**, occupancy **4 → 5**. Code shrank as expected.

**Timing (3-run avg)**: 48.683 s vs exp06 46.352 s = **+5.0 % slower** — the same occupancy-cliff regression we saw in exp02/03.

**Correctness**: bit-identical.

**Decision**: **Reverted**. Counted as unsuccessful **1/10** since the exp06 reset.

---

### Experiment 08 — Fast-math + force occupancy=4 via `__launch_bounds__(256,2)` + `amdgpu_waves_per_eu(4,4)`

**Hypothesis**: pin occupancy at 4 (sweet spot) so the fast-math VGPR savings can be reinvested in software pipelining / longer scheduling rather than wasted on a counterproductive occupancy bump.

**Change**: added the two attribute decorations to the kernel; kept fast-math from exp07.

**Assembly metrics**: VGPRs 89, **occupancy forced back to 4**. Behaviour as designed.

**Timing (3-run avg)**: 46.448 s vs exp06 46.352 s = **+0.21 %** (essentially tied / very slight regression).

**Correctness**: bit-identical.

**Why no win**: with occupancy capped at 4, the compiler had every chance to reuse the freed VGPRs. It did not (the saved VGPRs from fast-math are not directly translatable into software pipelining for this kernel — the bottleneck is memory latency on the indirect gathers, not compute scheduling).

**Decision**: **Reverted** — fast-math + attribute. Counted as unsuccessful **2/10**.

---

### Experiment 09 — **Promote `xout_zone` to LDS** ✅ SECOND SUCCESS

**Hypothesis**: After exp06 (rate → LDS), the next biggest source of redundant global memory traffic is `xout_zone[reactant_X[r]]` in the flux loop. Each thread does up to 3 gathers per reaction × NUM_REACTIONS / blockDim.x reactions per timestep × ~1000 timesteps per zone — millions of indirect global loads. `xout_zone` is only `SIZE` doubles (1.2 KB at SIZE=150), so it fits trivially in LDS.

**Change**:

- Kernel: added `double* xout_lds = rate + NUM_REACTIONS` to the LDS layout; aliased `xout_zone = xout_lds` inside the per-zone loop; added an explicit copy-back loop at the end of each zone iteration to write `xout_lds[i] -> xout[zone*SIZE + i]` for the host.
- Launcher: `sharedmem_allocation += SIZE * sizeof(double)` (only +1.2 KB).

**Assembly metrics**:

| Metric    | exp06 |  exp09 |  Δ |
| --------- | ----: | -----: | -: |
| VGPRs     |   105 |    110 | +5 |
| SGPRs     |   100 |    100 |  0 |
| Occupancy |     4 |      4 |  0 |

**LDS budget**: ~26.8 KB/block now (still 2 blocks/CU with 64 KB/CU).

**Timing (3-run avg, C55/C56/C57)**:

| metric         |       exp06 |       exp09 |       Δ |
| -------------- | ----------: | ----------: | ------: |
| avg_gpu_time   |   46.352 s  |   46.013 s  | **−0.73 %** |
| vs baseline    |             |             | **−1.43 %** cumulative |

Per-node: C55 45.941, C56 46.167, C57 45.930.

**Correctness**: full output diff against baseline shows ONLY the timing-line and cycle-count differ; every numerical line (xout, sdot, etc.) is **bit-identical**.

**Why it works**: every thread now reads `xout_zone[idx]` from LDS instead of from global. LDS reads are ~10× lower latency, in cache-equivalent of L1, and don't compete with the remaining indirect global gather (`flux[f_plus_map[j]]`). The species-update loop also writes `xout_zone[i]` to LDS instead of global, removing those store-to-cache stalls. The single global writeback at the end (`xout[zone*SIZE+i] = xout_lds[i]`) is fully coalesced and amortized over the entire while-loop's worth of work.

**Decision**: **KEPT**. Failure counter reset 2 → 0.

---

### Experiment 19 — **Promote `f_plus_factor[]` and `f_minus_factor[]` (uchar) to LDS** ✅✅✅ LARGEST WIN

**Hypothesis**: After exp15 packed `f_*_map` into LDS, the j-loops in the species-update phase still issue one global double load per iteration for `f_plus_factor[j]` / `f_minus_factor[j]`. Each per-zone integration runs ~1000 timesteps, and each timestep walks the entire flux-factor arrays (NUM_FLUXES_PLUS=2710 + NUM_FLUXES_MINUS=2704 ≈ 5414 doubles ≈ 43 KB total) — well above the per-CU L1 (~16 KB) and L2 budget shared with the rate prologue. That implies ~564 M global double loads per run just for those two arrays, and these are the ONLY remaining inputs in the j-loops still served from global memory. Critically, the values are stoichiometric coefficients (the `(double)reaction_mask[i][j]` cast in `parser.c:423`), all small non-negative integers (typically 1, 2, 3) — `(double)uchar` round-trips exactly in IEEE-754, so packing them as `unsigned char` is bit-preserving.

**Change**:

- Kernel: added `unsigned char* fp_fac_lds`, `unsigned char* fm_fac_lds` to the LDS layout (NUM_FLUXES_PLUS + NUM_FLUXES_MINUS = 5414 bytes ≈ 5.3 KB), staged from global once per block, and rewrote the j-loop multiplications as `(double)fp_fac_lds[j] * flux[(int)fp_map_lds[j]]` (and the symmetric minus form).
- Launcher: added `(NUM_FLUXES_PLUS + NUM_FLUXES_MINUS) * sizeof(unsigned char)` to `sharedmem_allocation`.

**Assembly metrics**:

| Metric    | exp17 |  exp19 |   Δ  |
| --------- | ----: | -----: | ---: |
| VGPRs     |   110 |    125 |  +15 |
| SGPRs     |   100 |    100 |    0 |
| Occupancy |     4 |      4 |    0 |

VGPR went **up** by 15 (likely extra base pointers / cast-and-extend code held live across the j-loops), but occupancy is unchanged at 4 waves/EU because we are still well above the 96-VGPR cliff.

**LDS budget**: ~51 KB/block (well within 64 KB/CU).

**Timing (6-run avg across nodes C55–C57, C59–C61)**:

| metric       |     exp17 |     exp19 |       Δ |
| ------------ | --------: | --------: | ------: |
| avg_gpu_time | 42.567 s  | **26.528 s** | **−37.7 %** vs exp17 |
| vs baseline  |           |           | **−43.2 %** cumulative |

Per-node (1 sample each): C55 26.535, C56 26.546, C57 26.488, C59 26.505, C60 26.586, C61 26.511 — extremely tight (σ ≈ 0.034 s).

**Correctness**: full output diff against baseline shows only the 8 expected timing/cycle-count lines differ; every `xout` and `sdot` line is **bit-identical**.

**Why it works (this is the dominant memory bottleneck)**:

The j-loop is the inner-most hot path of the entire kernel. After exp15 it looked like:

```
for (int j = p0; j <= p1; ++j)
    plus += f_plus_factor[j] * flux[(int)fp_map_lds[j]];
```

with two LDS reads (`fp_map_lds[j]`, `flux[idx]`) and one global double read (`f_plus_factor[j]`). Order-of-magnitude estimate of global-memory pressure for that one line:

- 104 zones × ~1000 timesteps × 2710 j-iters ≈ 282 M doubles read for `f_plus_factor` alone (×8 B = 2.25 GB), plus the symmetric ~280 M for `f_minus_factor` → ~4.5 GB of global traffic per outer iteration.
- The `f_*_factor` arrays together are 43 KB — too large for L1 (~16 KB), forcing repeated L2 (and partial HBM) refills for every species-i pass.

Moving them into 5.3 KB of LDS:

1. Eliminates ~4.5 GB of HBM/L2 read traffic per outer iteration outright.
2. Removes the `s_waitcnt vmcnt(0)` stalls that previously bracketed each j-iteration's global load (the j-loop is now a pure LDS pipeline of three LDS reads and an FMA).
3. Frees L2 bandwidth and capacity for the still-global rate-prologue loads (`prefactor[]`, `p_0..6[]`), which run once per zone and don't benefit from LDS-residency at this LDS budget.

The fact that VGPR went up 110 → 125 with no occupancy change *and a 38 % runtime drop* is a textbook indicator that this kernel was completely memory-latency-bound on these two arrays — not VGPR-bound, not occupancy-bound, not compute-bound.

**Decision**: **KEPT**. Failure counter reset 3 → 0.

---

## Summary table (running)

| #  | Description                              | Avg GPU time (1 iter, 104 zones) |  Δ vs baseline | VGPR | SGPR | Scratch | Occ | Decision |
| -- | ---------------------------------------- | -------------------------------: | -------------: | ---: | ---: | ------: | --: | -------- |
| 00 | baseline                                 |                        46.7017 s |             —  |  117 |  100 |       0 |   4 | —        |
| 01 | `pow(tmp,1/3)` → `cbrt(tmp)`             |                        46.7434 s |       +0.09 %  |  103 |  100 |       0 |   4 | kept (cleanup; unsuccessful 1/10) |
| 02 | remove `isfinite(f_rate)` printf guard   |          48.7429 s (3-run avg)   |       +4.36 %  |   93 |  100 |       0 |   5 | **reverted** (unsuccessful 2/10) — occupancy 4→5 hurt due to cache contention |
| 03 | `__restrict__` on all pointer params     |          49.170  s (3-run avg)   |       +5.33 %  |  103 |  100 |       0 |   4 | **reverted** (unsuccessful 3/10) — compiler re-sched hurt latency hiding |
| 04 | cache `num_react_species[r]` in local    |          46.737  s (3-run avg)   |       +0.12 %  |  103 |  100 |       0 |   4 | reverted (unsuccessful 4/10) — compiler already CSE'd |
| 05 | blockdim 256 → 128                       |          50.563  s (3-run avg)   |       +8.33 %  |  104 |  100 |       0 |   4 | **reverted** (unsuccessful 5/10) — per-thread work doubled |
| 06 | **`rate[]` → LDS** ✅                     |          46.352  s (3-run avg)   |       **−0.70 %**  |  105 |  100 |       0 |   4 | **KEPT** — first success |
| 07 | scoped fast-math (`-fno-math-errno -ffinite-math-only -fno-signed-zeros`) | 48.683 s (3-run avg)   |   +5.00 % vs exp06 |   89 |  100 |       0 |   5 | **reverted** (unsuccessful 1/10 since reset) — occupancy cliff again |
| 08 | exp07 + force occ=4 (`__launch_bounds__(256,2)` + `amdgpu_waves_per_eu(4,4)`) | 46.448 s (3-run avg) | +0.21 % vs exp06 |   89 |  100 |       0 |   4 | **reverted** (unsuccessful 2/10) — fast-math savings unused |
| 09 | **`xout_zone` → LDS** ✅                  |          46.013  s (3-run avg)   |       **−0.73 % vs exp06**, −1.43 % cumulative |  110 |  100 |       0 |   4 | **KEPT** — second success |
| 10 | **`num_react_species` packed as `uchar` → LDS** ✅ |          45.647  s (3-run avg)   |       **−0.80 % vs exp09**, −2.21 % cumulative |  110 |  100 |       0 |   4 | **KEPT** — third success |
| 11 | **`reactant_1` (uchar) → LDS** ✅          |          45.256  s (3-run avg)   |       **−0.86 %**, −3.05 % cumulative |  110 |  100 |       0 |   4 | **KEPT** |
| 12 | **`reactant_2` (uchar) → LDS** ✅          |          45.073  s (3-run avg)   |       **−0.40 %**, −3.44 % cumulative |  110 |  100 |       0 |   4 | **KEPT** |
| 13 | **`reactant_3` (uchar) → LDS** ✅          |          44.881  s (3-run avg)   |       **−0.43 %**, −3.85 % cumulative |  110 |  100 |       0 |   4 | **KEPT** |
| 14 | **`f_plus_max` / `f_minus_max` (ushort) → LDS + hoist bound** ✅ |          44.733  s (3-run avg)   |       **−0.33 %**, −4.17 % cumulative |  110 |  100 |       0 |   4 | **KEPT** |
| 15 | **`f_plus_map` / `f_minus_map` (ushort) → LDS** ✅✅ |          42.652  s (3-run avg)   |       **−4.65 % vs exp14**, **−8.63 % cumulative** |  110 |  100 |       0 |   4 | **KEPT** — biggest single win |
| 16 | `aa[SIZE]` → LDS                          |          42.636  s (3-run avg)   |       −0.04 % (within noise) |  110 |  100 |       0 |   4 | kept (cleanup; counted as unsuccessful 1/10 since reset) |
| 17 | cache `xout_zone[i]` in register in species update | 42.567 s (6-run avg)     |       −0.16 % vs exp16 (within noise) |  110 |  100 |   0 |   4 | kept (cleanup; counted as unsuccessful 2/10) |
| 18 | `q_value[]` → LDS (double)                |          42.517  s (6-run avg)   |       −0.12 % vs exp17 (within noise) |  123 |  100 |       0 |   4 | **reverted** — VGPRs jumped 110→123 with no benefit (unsuccessful 3/10) |
| 19 | **`f_plus_factor` / `f_minus_factor` (uchar) → LDS** ✅✅✅ | **26.528 s (6-run avg)** | **−37.7 % vs exp17, −43.2 % cumulative** | 125 | 100 | 0 | 4 | **KEPT** — biggest single win by far |

## Failure counter

Consecutive unsuccessful experiments: **0 / 10** (reset after exp19 success)

Failure log (so far):

- exp01: cbrt — bit-identical, no measurable speedup (kept as cleanup). _unsuccessful 1/10._
- exp02: remove isfinite — reproducibly slower (+4.4%); reverted. _unsuccessful 2/10._
- exp03: `__restrict__` — reproducibly slower (+5.3%); reverted. _unsuccessful 3/10._
- exp04: cache `num_react_species[r]` — compiler already CSE'd. _unsuccessful 4/10._
- exp05: blockdim 128 — reproducibly slower (+8.3%). _unsuccessful 5/10._
- **exp06: rate → LDS — −0.70 % win. Counter reset.**
- exp07: fast-math — +5.0 % vs exp06 (occupancy cliff). _unsuccessful 1/10._
- exp08: fast-math + force occ=4 — +0.2 %. _unsuccessful 2/10._
- **exp09: xout_zone → LDS — −0.73 % vs exp06, −1.43 % cumulative. Counter reset.**
- **exp10: num_react_species (uchar) → LDS — −0.80 % vs exp09, −2.21 % cumulative. Counter reset.**
- **exp11: reactant_1 (uchar) → LDS — −0.86 %, −3.05 % cumulative.**
- **exp12: reactant_2 (uchar) → LDS — −0.40 %, −3.44 % cumulative.**
- **exp13: reactant_3 (uchar) → LDS — −0.43 %, −3.85 % cumulative.**
- **exp14: f_plus_max / f_minus_max (ushort) → LDS + hoist bound — −0.33 %, −4.17 % cumulative.**
- **exp15: f_plus_map / f_minus_map (ushort) → LDS — −4.65 % vs exp14, −8.63 % cumulative. Largest single win.**
- exp16: aa[] → LDS — within noise (−0.04 %). Kept for cleanliness; counted as _unsuccessful 1/10_ since the post-exp15 reset.
- exp17: cache `xout_zone[i]` in register — within noise (−0.16 %). Kept for clarity; _unsuccessful 2/10._
- exp18: `q_value[]` → LDS — within noise (−0.12 %); VGPRs jumped 110→123. **Reverted.** _unsuccessful 3/10._
- **exp19: `f_plus_factor` / `f_minus_factor` (uchar) → LDS — −37.7 % vs exp17, −43.2 % cumulative. Largest single win in the entire study. Counter reset.**

### Pattern so far

The baseline configuration (4 waves/EU, ~103 VGPRs post-cbrt) is at a memory-latency-bound sweet spot. Changes that "look" helpful to the static compiler (higher occupancy, stronger aliasing info, shorter code) consistently hurt runtime. The one successful change (exp06) was a **data-placement** optimization that reduced global-memory traffic without touching occupancy.

Optimization focus going forward:

1. Continue to **reduce global-memory traffic** on the hot paths (e.g., pack metadata into LDS too).
2. **Shorter hot-path instruction sequences** that keep VGPR count above the 96-threshold so occupancy stays at 4.
3. Math-library simplifications in the prologue (already explored with exp01/cbrt).
4. Explore compiler flags that trim math expansions (fast-math) without altering control-flow structure.
