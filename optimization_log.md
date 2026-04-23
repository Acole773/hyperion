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

---

## Post-merge validation (Hackathon2026 tip)

After all optimization work was merged into `origin/Hackathon2026`, the merged
kernel was validated against the original baseline.

### What's merged

`origin/Hackathon2026` HEAD = `1c9eec9` includes, on top of `8d7366a`:

- `f1dee2e` — exp01, exp06, exp09–exp16 (LDS data placement, packed metadata).
- `01546d3` — exp17 + **exp19** (pack `f_plus_factor` / `f_minus_factor` as `uchar`
  in LDS).
- `571f634` — sentinel-based indexing for `f_*_max` / `f_*_min` (replaces the
  `(i == 0) ? 0 : f_*_max[i-1] + 1` conditional with a uniform
  `f_*_max[i] + 1`, by adding a leading sentinel element with values
  `-1` / `0`).
- Merge commit `5d36831` resolved conflicts in `bn_burner_gpu.hip` and
  `main_gpu.c` to combine the LDS work and the sentinel refactor.

### Run

| setting          | value                       |
|------------------|-----------------------------|
| build job        | `274424` (clean)            |
| run job          | `274425`                    |
| node             | MI250                       |
| zones, iterations| 104, 1                      |
| log              | `output_20260420_173757.txt`|

### Timing (avg over 1 iteration, single run)

| variant                                                                | avg_gpu_time | Δ vs baseline |
|------------------------------------------------------------------------|-------------:|--------------:|
| baseline (`8d7366a`)                                                   |    46.7017 s |             — |
| exp19 standalone (kernel-optimization-experiments tip, 6-run avg)      |    26.528  s |     **−43.2 %** |
| **merged Hackathon2026 (LDS + exp19 + sentinel), 1-run**               | **26.482 s** | **−43.30 %**  |

The sentinel refactor itself produces no measurable timing delta on top of
exp19 (within run-to-run noise), as expected — it removes a few SGPR-side
instructions in the hot loop but doesn't change the dominant memory pattern.

### Numerical correctness

`diff baseline_output.txt output_20260420_173757.txt` shows **only the
cycle-count line** differs:

```
< Total cycles per run of batch (avg over 1 iterations, rnded): 1100190166
> Total cycles per run of batch (avg over 1 iterations, rnded):  623866653
```

Every `xout[i]` and every `sdot[zone]` value is bit-identical to the baseline
across all 104 zones × 150 species and all 104 sdotrate values. This confirms
that the LDS uchar packing of `f_plus_factor` / `f_minus_factor` and the
sentinel-based loop bounds are both numerically lossless on this network.

### Status

The Hackathon2026 branch tip is **validated**: bit-identical results, ~43 %
end-to-end speedup vs the original baseline.

---

## Phase 2: divergence-focused experiments (post-merge)

After validating the merged Hackathon2026 tip, we ran `rocprof-compute` to
identify the new dominant bottleneck. With the LDS data-placement work
having essentially eliminated HBM/L2 traffic (L2-Fabric BW = 0 Gb/s, L2 hit
rate = 98 %), the kernel is no longer memory-bandwidth-limited.

### Phase-2 baseline metrics (Hackathon2026 tip, job 274426, 14-iter rocprof)

| metric                       | value           | comment |
|------------------------------|-----------------|---------|
| VALU Active Threads          | **9.59 / 64 (15 %)** | severe intra-wave divergence |
| VALU Utilization             | 5.33 %          | ALU mostly idle |
| SALU Utilization             | 1.49 %          | scalar pipe also idle |
| IPC                          | 0.12 / 5.0      | stalling (2.4 % of peak) |
| Wavefront Occupancy          | 410 / 3328 (12.3 %) | LDS-bound: 51 KB/block × 1 block/CU |
| Branch Utilization           | 0.75 %          | low *frequency*, but each branch diverges the wave |
| LDS Bank Conflicts/Access    | 0.13            | minimal |
| L2-Fabric Read/Write BW      | 0 Gb/s          | HBM traffic eliminated |
| L2 Cache Hit Rate            | 98.03 %         | great cache behavior |
| VGPRs / SGPRs / scratch / occ| 125 / 96 / 0 / 4 waves/SIMD | per-block resource use |

Diagnosis: the kernel is now **divergence-bound**, not memory- or compute-bound.
The 15 % VALU active-thread rate is the single biggest lever. LDS-driven
occupancy (1 block/CU) is secondary — would help, but cutting LDS below 32 KB
to fit 2 blocks/CU would force flux+rate (the two 12 KB doubles arrays) back
to global memory, which would likely hurt more than it helps.

We adopted the same convention as Phase 1: 10 unsuccessful attempts in a row
stops the loop. Timing methodology tightened to **3-iter runs** (instead of
1-iter) for noise reduction, since differences at this scale (~26 s baseline)
are now in the 0.2–1 % range.

### Experiment 21 — Branchless flux loop via {0,1} filter

**Intent**: eliminate the divergent `if (nrs > 1)` / `if (nrs > 2)` chain in
the flux computation.

**Change**:

- `parser.c`: switched `reactant_idx[n]` to `calloc` so unused reactant slots
  are zero-initialized (previously uninitialized memory; the kernel only
  read them when `nrs > k`, so the bug was masked by the conditional).
- `bn_burner_gpu.hip`: replaced

  ```c
  if (nrs > 1) f *= xout_zone[r2];
  if (nrs > 2) f *= xout_zone[r3];
  ```

  with branchless

  ```c
  double f2 = (double)(nrs > 1);
  double f3 = (double)(nrs > 2);
  double term2 = f2 * a2 + (1.0 - f2);   // 1.0 if unused, a2 otherwise
  double term3 = f3 * a3 + (1.0 - f3);
  flux[r] = rate[r] * a1 * term2 * term3;
  ```

  Also removed the `isfinite(rate)` printf debug guard (another divergent
  branch executed every reaction).

**Timing (104 zones, 3 iter, single run)**: 26.329 s vs Hackathon2026 tip
26.482 s (1-iter) — at the edge of noise.

**Numerics**: bit-identical xout / sdot vs baseline.

**Decision**: kept (and superseded by exp25, see below).

### Experiment 22 — Branchless asymptotic-vs-Euler in species update

**Intent**: eliminate the divergent `if (minus * dt > x_i) ... else ...`
branch in the species-update i-loop.

**Change**: compute both candidates and select with a ternary:

```c
double asym = (x_i + plus*dt) / (1.0 + (minus / (x_i + ZERO_FIX)) * dt);
double eul  = x_i + (plus - minus)*dt;
x_i = (minus*dt > x_i) ? asym : eul;
x_i = fmax(x_i, ZERO_FIX);
```

**Timing (104 zones, 3 iter)**: 26.531 s vs E21 26.329 s — slight
regression (+0.77 %).

**Numerics**: bit-identical.

**Decision**: **reverted**. The extra `v_div_f64` (~30 cycles) on lanes that
would have taken the cheaper Euler path outweighs the divergence saved on a
low-frequency branch (the species-update i-loop runs only ~1 iteration per
thread per timestep at SIZE=150, blockDim=256). Counter: **1/10**.

### Experiment 23 — blockDim 256 → 192 (skipped)

**Intent**: better match SIZE=150 (with 256 threads, 106 lanes idle in the
species-update i-loop; with 192, only 42 lanes idle).

**Why skipped without measuring**:

1. The cross-wave reduction at the end of the kernel uses a power-of-2
   tree: `for (int offset = num_waves >> 1; offset > 0; offset >>= 1)`.
   With blockDim=192, num_waves=3 (not power-of-2): the loop runs once
   (offset=1) leaving wave-2's partial sum uncombined. **Bug.**
2. The kernel is already LDS-occupancy-bound at 1 block/CU (51 KB/block,
   64 KB/CU). Reducing blockDim from 256 → 192 doesn't change LDS use,
   but cuts wavefronts/CU from 4 → 3 (lower latency hiding).

A future pass could try blockDim=128 (2 waves, power-of-2) but the same
LDS-bound occupancy concern applies.

### Experiment 25 — Branchless flux via dummy-slot trick (refinement of E21)

**Intent**: drop the {0,1} filter math from E21 by exploiting the fact that
`a * 1.0 == a`. Multiply by 1.0 unconditionally for unused reactants.

**Change**:

- `parser.c`: a new `rate_library_fixup_unused_reactants()` (called from
  `init.c` *after* both `rate_library_create` and `network_create`, so that
  `num_species` is set) replaces unused reactant slots with sentinel index
  `num_species`.
- `bn_burner_gpu.hip`: extend `xout_lds` from `SIZE` to `SIZE + 1`. After
  the per-zone init, thread 0 sets `xout_zone[SIZE] = 1.0`. The flux loop
  becomes:

  ```c
  double a1 = xout_zone[idx1];
  double a2 = xout_zone[idx2];   // = 1.0 if unused
  double a3 = xout_zone[idx3];   // = 1.0 if unused
  flux[r] = rate[r] * a1 * a2 * a3;
  ```

  No filter math, no `nrs_lds` load in the flux loop, no int→double
  conversions.
- `bn_burner_gpu.c`: bumped `sharedmem_allocation` by `sizeof(double)` for
  the dummy slot (51,374 B → 51,382 B, no occupancy impact).

**Initial bug found and fixed**: the first cut put the parser fixup *inside*
`rate_library_create`, but `num_species` is only set later by
`network_create`. The unused slots were therefore set to `0` (a real
species), corrupting the flux computation (sdot = 3.83e-32 vs 6.09e+15).
Fixed by promoting the fixup to a separate function called from `init.c`
after both create routines run.

**Timing (104 zones, 3 iter, average of 3 runs)**: **26.477 s ± 0.004**
vs E21 26.329 s (1 run). Effectively the same as E21 within noise.

**Numerics**: bit-identical xout / sdot vs baseline.

**Assembly**: VGPRs **125 → 118** (−7), SGPRs 96 → 100 (+4), scratch 0,
occupancy still 4 waves/SIMD. The `nrs_lds` load + 4 fma + 2 cvt removal
shows up as the VGPR drop.

**Decision**: kept as the canonical branchless-flux form (cleaner code,
fewer instructions, fixes the parser uninitialized-memory issue).
Counter: **2/10** (no measurable timing improvement on its own).

### Experiment 26 — Sort species by j-loop length (intra-wave divergence)

**Intent**: directly attack the dominant remaining bottleneck identified by
the post-merge `rocprof-compute` (VALU active threads ~15 %). The flux
loops were already touched by E21/E25 with no measurable effect, so the
divergence is now *inside the species-update i-loop*, where each thread
walks a `[p0,p1]` and `[m0,m1]` flux range whose length varies wildly per
species (the network includes a few species with hundreds of contributing
fluxes alongside many with only one or two). Within a wavefront the lanes
all wait for the longest-j lane.

**Hypothesis**: if we sort species by total j-length (plus + minus) in
descending order and process them in that order, lanes inside any one
wavefront see j-lengths that are close to each other, so per-wave
max-j ≈ per-wave avg-j. Total work is identical; only intra-wave waiting
shrinks.

**Change**:

- `src/parse-data/parser.c`: new `compute_species_perm()` allocates
  `species_perm[num_species]` and fills it with species indices sorted by
  `(f_plus_max[i+1] - f_plus_max[i]) + (f_minus_max[i+1] - f_minus_max[i])`
  descending (insertion sort — `num_species` ≤ 365). Called from
  `init.c` *after* `data_init()` so `f_plus_max` / `f_minus_max` are
  populated.
- `src/core/store.{h,c}`: `extern int* species_perm;` declared/defined as
  global.
- `src/hipcore/bn_burner_gpu.h`: added `int* species_perm` to
  `burner_args_t` and to the kernel signature.
- `src/hipcore/bn_burner_gpu.c`: `hipMalloc` + copy + free for
  `args.species_perm`; pass to kernel; bumped `sharedmem_allocation` by
  `SIZE * sizeof(unsigned char)` (= 150 B at SIZE=150, fits in uchar
  since species idx < SIZE).
- `src/hipcore/bn_burner_gpu.hip`: stage `species_perm` into LDS as
  `perm_lds[SIZE]` (uchar). Species-update loop:
  `for (int t = tid; t < SIZE; t += blockDim.x) { int i = perm_lds[t]; ... }`
  Only the species-indexed reads (`fp_max_lds[i]`, `fm_max_lds[i]`,
  `xout_zone[i]`) use the permuted index; everything inside the j-loops
  stays reaction-indexed and is unaffected.

**Timing (104 zones, 3 iter, 3 runs across C59/C62)**:

| run | node | avg_gpu_time |
|-----|------|-------------:|
| 275656 | C59 | 26.535896 s |
| 275657 | C59 | 26.540372 s |
| 275658 | C62 | 26.516714 s |

**Average: 26.531 s** vs E25 baseline 26.477 ± 0.004 s — **+0.20 % regression**
(σ ≈ 0.013, so the regression is statistically real but tiny).

**Numerics**: bit-identical `xout` and `sdot` vs baseline.

**Why it didn't help (post-mortem)**:

1. With blockDim=256 and SIZE=150, the species-update i-loop runs **at most
   1 pass per thread** (lanes 150–255 always idle). So each wave's runtime is
   dominated by the *single longest j-loop assigned to one of its lanes*, not
   by intra-wave divergence among multiple j-loop iterations per lane.
2. The MI250 wave scheduler interleaves divergent paths via `EXEC` masking;
   reordering species-to-lane assignment doesn't change the total cycles
   spent per-wave when the longest-j lane dominates.
3. The extra `perm_lds[t]` LDS load has 1 cycle of latency that is fully
   hidden, so it doesn't add cost — but it also adds no benefit.
4. The actual bottleneck appears to be the **per-species j-loop length
   itself** (the longest species has hundreds of j-iterations, ~10× the
   median), which sorting cannot fix — the longest-j species is one lane
   in *some* wave regardless of ordering.

**Decision**: **REVERTED**. Code, host structures, and global declarations
removed. Counter: **3/10**.

**Lesson for next experiments**: to actually move the VALU-active-threads
needle, we need to break up the long j-loops across multiple threads — not
reassign whole species to threads. That points to a cooperative atomic
accumulation pattern (E28).

### Experiment 27 — Use `nrs_lds` (uchar LDS) in rate-eval loop instead of global `num_react_species`

**Intent**: small cleanup — the rate-eval loop was reading
`num_react_species[r]` from global memory even though the LDS-resident
`nrs_lds[r]` (staged once per block, exp10) carries the same value.
Replacing the global load with an LDS load saves NUM_REACTIONS × Nzones
≈ 167 K HBM loads per outer iteration.

**Change**: in `bn_burner_gpu.hip`, the rate-eval loop now does
`int ns = (int)nrs_lds[r] - 1;` instead of
`int ns = num_react_species[r] - 1;`.

**Timing (104 zones, 3 iter, 3 runs)**:

| run | node | avg_gpu_time |
|-----|------|-------------:|
| 275661 | C59 | 26.486732 s |
| 275662 | C62 | 26.474932 s |
| 275663 | C59 | 26.481966 s |

**Average: 26.481 s** vs E25 26.477 ± 0.004 s — within noise (+0.015 %).

**Numerics**: bit-identical (`diff` of `sdot`/`xout` lines = 0).

**Decision**: **kept** as cleanup (LDS load is cleaner than HBM load, no
runtime cost). Counter: **4/10** (no measurable speedup).

### Experiment 28 — Wave-cooperative species update (parallelize the j-loops) ✅✅✅ HUGE WIN

**Intent**: kill the dominant divergence source identified by rocprof
(VALU active threads ~15 %). The pre-E28 species-update loop assigned
one species to each thread; lanes within a wavefront then walked
wildly different j-loop lengths sequentially, so the wave waited for
its longest-j lane every timestep. With blockDim=256 / SIZE=150 only
150 of 256 lanes are even active here, compounding the issue.

**Hypothesis**: instead of "1 thread per species, sequential j-loop",
do "1 wave per species, parallel j-loop with shfl reduction". Each wave
handles SIZE/num_waves species sequentially (~38 species per wave at
blockDim=256), but each species' j-loop is split across 64 lanes with a
warp-shuffle reduction. The per-wave critical path now scales with
*sum-of-j-counts / 64* instead of *max-j-count*, and all 64 lanes per
wave do useful work.

**Change** (kernel only, no host changes, no LDS budget change):

```c
const int wave = tid >> 6;
const int lane = tid & 63;
const int nw   = blockDim.x >> 6;

for (int i = wave; i < SIZE; i += nw) {
    int p0 = (int)fp_max_lds[i] + 1;
    int p1 = (int)fp_max_lds[i + 1];
    int m0 = (int)fm_max_lds[i] + 1;
    int m1 = (int)fm_max_lds[i + 1];

    double plus = 0.0;
    for (int j = p0 + lane; j <= p1; j += 64)
        plus += (double)fp_fac_lds[j] * flux[(int)fp_map_lds[j]];
    for (int offset = 32; offset > 0; offset >>= 1)
        plus += __shfl_down(plus, offset, 64);

    double minus = 0.0;
    for (int j = m0 + lane; j <= m1; j += 64)
        minus += (double)fm_fac_lds[j] * flux[(int)fm_map_lds[j]];
    for (int offset = 32; offset > 0; offset >>= 1)
        minus += __shfl_down(minus, offset, 64);

    if (lane == 0) {
        double x_i = xout_zone[i];
        if (minus * dt > x_i) {
            x_i = (x_i + plus * dt) /
                  (1.0 + (minus / (x_i + ZERO_FIX)) * dt);
        } else {
            x_i += (plus - minus) * dt;
        }
        if (x_i < ZERO_FIX) x_i = ZERO_FIX;
        xout_zone[i] = x_i;
    }
}
```

**Timing (104 zones, 3 iter, 2 runs across C59/C62)**:

| run | node | avg_gpu_time |
|-----|------|-------------:|
| 275665 | C59 | 13.241490 s |
| 275666 | C62 | 13.235267 s |

**Average: 13.238 s** vs E27 26.481 s → **−50.0 %** in this experiment alone.
vs Phase-1 final E19 26.528 s → **−50.1 %**.
vs original baseline 46.7017 s → **−71.6 % CUMULATIVE**.

**Numerics**: **bit-identical** to baseline (sdot max rel err = 0.0 across
all 104 zones, xout max rel err = 0.0 across all 150 species). The
parallel-reduction reordering happened to not move any bits at this
precision for this network — much better than the 1e-12 tolerance limit.

**Why it works**: the species-update is the inner-loop's compute-heavy
phase, dominated by LDS reads and FMAs in the j-loops. By spreading each
species' j-iterations across 64 lanes:

1. Per-species j-work parallelizes 64×.
2. The wave's critical path is now `(sum_j / 64) + log2(64) shfl cycles`
   ≈ `36/64 + 6 = ~7 cycles` for an average species, vs `200 cycles`
   for the longest species in the old form.
3. Inactive lanes vanish: all 256 threads do useful work (4 waves × 64
   lanes, every species update has all 64 lanes contributing).
4. The asym/Euler scalar update (with its divide and branch) only runs
   on `lane == 0`, so the divergence-cost of the rare asym branch is
   amortized over a wave instead of paid per-thread.

**Decision**: **KEPT**. Largest single-experiment win since E19. Failure
counter reset 4 → 0.

**Assembly (`asm_e28/e28.s`)**: VGPRs 116 (E25 was 118), SGPRs 100,
scratch 0. The wave-cooperative form actually *reduced* VGPR pressure
slightly because each thread no longer needs to hold the full `plus`
and `minus` accumulators across the entire (potentially long) j-loop —
the partial sums only live across 64-strided iterations.

**rocprof-compute (job 275667, 14-iter, kernel-filtered)**:

| metric                       | pre-E28 (E25) | **E28**      | Δ |
|------------------------------|--------------:|-------------:|---:|
| VALU Active Threads          | 9.59/64 (15 %)| **29.3/64 (45.78 %)** | **+206 %** |
| VALU Utilization             | 5.33 %        | 10.42 %      | +95 % |
| IPC                          | 0.12          | **0.35**     | +192 % |
| IPC (Issued)                 | —             | 0.83         | — |
| Wavefront Occupancy          | 410/3328 (12.3 %) | 299.5/3328 (9.0 %) | −27 % |
| LDS Bank Conflicts/Access    | 0.13          | **0.03**     | −77 % |
| Branch Utilization           | 0.75 %        | 1.21 %       | +61 % |
| Active CUs                   | —             | 82/104 (78.8 %) | — |
| VGPRs / SGPRs / scratch      | 125 / 96 / 0  | 116 / 112 / 0 | VGPR −9 |
| L2 Cache Hit Rate            | 98.0 %        | 98.24 %      | unchanged |
| L2-Fabric Read/Write BW      | 0 Gb/s        | 0 Gb/s       | unchanged |

**Diagnosis after E28**:

1. Divergence is still *the* bottleneck (45.78 % active threads, not 100 %),
   but it's no longer the *dominant* bottleneck. The remaining 54 % of
   inactive lanes come from:
   - The `if (lane == 0)` scalar asym/Euler update at the end of each
     species iteration: 63 of 64 lanes idle for ~30 cycles per species
     (longer when the asym branch is taken because of the divide).
   - Tail effects when a species' j-count is not a multiple of 64.
2. Wavefront occupancy *decreased* slightly (12.3 % → 9.0 %) because
   SGPR count rose 96 → 112 (adds 1 SGPR-bank pressure but no
   functional change). Still LDS-bound at 1 block/CU; the SGPR bump
   doesn't change the binding constraint. Total runtime dropped 50 %
   despite this, so the per-wave throughput is what matters here, not
   occupancy.
3. Active CUs at 78.8 % suggests imperfect zone→CU distribution. With
   104 zones × gridDim=zones on an MI250 GCD that has 104 CUs, we
   would expect 100 %; the gap likely reflects the fact that an MI250
   has 110 physical CUs per GCD with only 104 enabled, plus
   wavefront-launch latency on the first/last few CUs.
4. **Memory subsystem is essentially perfect** (0 Gb/s HBM, 98 % L2
   hit). All future experiments must focus on either compute or
   divergence — *not* memory placement (already won by E06–E19).
5. The rate-eval loop and the energy/norm reductions are now a larger
   fraction of total runtime. Measured cycles in those phases haven't
   changed but the species update is now ~2× faster, so amortization
   has shifted.

**Cumulative state at end of E28**:

| metric                | original baseline | **E28**     | Δ |
|-----------------------|------------------:|------------:|---:|
| avg_gpu_time (104z, 1it) | 46.7017 s     | **~13.24 s** | **−71.6 %** |
| VGPRs                 | 117              | 116        | −1 |
| Wavefront occupancy   | 4 waves/SIMD     | 4 waves/SIMD | unchanged |
| L2-Fabric BW          | high (memory-bound) | 0 Gb/s    | eliminated |
| VALU active threads   | 21.6 % (early)   | 45.78 %    | +112 % |


### Experiment 29 — Two-phase species update (Phase A reduce / Phase B asym-Euler) — **REVERTED**

**Intent**: kill the remaining `if (lane == 0)` scalar asym/Euler
serialization identified in the E28 rocprof. In E28 each wave does ~38
species sequentially and burns ~30 cycles per species on lane 0 (with
63 idle lanes) for the asym/Euler scalar update. Plan: split into

- **Phase A** — wave-cooperative j-loop reduction (same as E28). Lane 0
  stages `plus`/`minus` to two new per-species LDS arrays
  `plus_lds[i]` / `minus_lds[i]`.
- **Phase B** — every thread (or every lane in its own wave) does one
  species' asym/Euler scalar update in parallel.

Two variants tried:

- **E29a (cross-wave Phase B)**: `for (int i = tid; i < SIZE; i += blockDim.x)`,
  with a `__syncthreads()` between phases (because thread `t` in wave
  `t/64` may read a species written by a different wave).
- **E29b (per-wave Phase B)**: lane `k` of wave `w` reads species
  `w + k * num_waves` (the k-th species this same wave produced in
  Phase A). No cross-wave dependency, so the block-level barrier is
  replaced by an inline `s_waitcnt lgkmcnt(0)` that just drains this
  wave's pending LDS stores.

**LDS cost**: +2 × SIZE × 8 = +2400 B (well under the 64 KB CU budget).

**Numerics**: bit-identical to baseline in both variants.

**Timing (104 zones, 2 iter, 2 runs each)**:

| variant | run | avg per-iter | cycles/zone-batch |
|---------|-----|-------------:|------------------:|
| E29a    | 275671 | 14.1254 s | 332 764 246 |
| E29a    | 275672 | 14.1222 s | 332 684 297 |
| E29b    | 275674 | 14.1394 s | 333 094 670 |
| E29b    | 275675 | 14.1346 s | 332 978 887 |
| **E28** (re-measured today, 275677/275678) | — | **13.240 s** | **311 898 406** |

**Result**: **+6.7 % regression** in both variants vs E28 (re-measured
on the same nodes the same day to rule out node variance — E28 came in
at 13.240 s, well within ±0.01 s of its original 13.238 s number). The
hypothesis was wrong:

- The lane-0 scalar update doesn't actually stall the wave for ~30
  cycles per species — on AMDGPU the scalar/VALU instructions still
  issue at 1 inst/cycle even with EXEC = 1 lane, and they overlap with
  the next wave's reduction work via the SIMD scheduler. So the "wasted
  63 lanes" are accounting fiction, not real critical path.
- What *is* real cost: the extra LDS round-trip (lane-0 store of
  `plus`/`minus` + per-lane reload in Phase B) and the synchronization
  between phases (block barrier in E29a, wave-local waitcnt in E29b).
  Both variants pay it, both regress identically (~+6.7 %).
- The per-wave variant (E29b) does drop the block-level barrier but
  still has to drain the LDS write queue and re-issue 2 LDS reads per
  species — apparently the LDS-RAW dependency cost dominates the
  barrier cost in the original layout.

**Decision**: **REVERTED**. The asym/Euler scalar work on lane 0 is
not actually a bottleneck at this point. Failure counter 0 → 1.

**Lesson**: don't trust "% inactive lanes" as a proxy for cost when
the scalar workload per wave is small and the wave scheduler can
already overlap it with other waves' compute. The real bottleneck must
be measured in *wave critical-path cycles*, not lane utilization.


### Experiment 30 — Increase blockDim 256 → 1024 (4 waves → 16 waves per block)

**Intent**: the E28 wave-cooperative species update has per-wave
critical path ≈ `(sum_of_j_counts_for_that_wave's_species) / 64 +
shfl_overhead × num_species_per_wave`. With blockDim=256 (4 waves),
each wave handles ~38 species, so the critical path per timestep is
dominated by ~38 × (avg_j/64 + 6 shfl ≈ 22) ≈ 1300 cycles. Going to
**16 waves per block** drops it to ~10 species per wave → ~340 cycles,
plus 4× more in-flight waves to hide LDS/HBM latency. LDS budget per
block is unchanged (LDS is block-shared, not per-wave). The kernel
source is unchanged — `blockDim.x` is a runtime parameter and `nw =
blockDim.x >> 6` already adapts the wave-cooperative loop bounds.

**Change**:

```c
// src/hipcore/bn_burner_gpu.c
- dim3 blockdim(256, 1, 1);
+ dim3 blockdim(1024, 1, 1);
```

A 512-thread (8-wave) intermediate point was also measured to chart the
trend.

**Timing (104 zones, 2 iter, 2 runs across two MI250 nodes)**:

| blockDim | waves/block | run     | avg per-iter | cycles/zone-batch |
|---------:|------------:|---------|-------------:|------------------:|
| 256 (E28)| 4           | 275677  | 13.2419 s    | 311 950 825 |
| 256 (E28)| 4           | 275678  | 13.2376 s    | 311 845 987 |
| 512      | 8           | 275680  |  7.5461 s    | 177 769 678 |
| 512      | 8           | 275681  |  7.5457 s    | 177 759 849 |
| **1024** | **16**      | 275683  | **6.4285 s** | **151 442 629** |
| **1024** | **16**      | 275684  | **6.4272 s** | **151 411 335** |

**Average E30 (1024)**: **6.428 s**, vs E28 13.240 s → **−51.5 %** in this
experiment alone. vs original baseline 46.7017 s → **−86.2 % CUMULATIVE**.

(The 8-wave point at 7.546 s gives −43.0 % vs E28; doubling once more to
16 waves gives another −14.8 %, showing diminishing returns as the
per-wave critical path stops dominating and the per-block fixed cost
starts to matter. 1024 is the gfx90a hardware max; cannot go higher.)

**Numerics**: **bit-identical** to baseline at both 512 and 1024 (sdot
diff vs `baseline_output.txt` = 0 lines for both runs).

**Assembly (`asm_e30/e30.s`)**: 28 823 lines, **identical** to
`asm_e28/e28.s` (same VGPR=115, SGPR=100, scratch=0). The kernel code
is unchanged — the entire win is from launch configuration, not
codegen.

**rocprof-compute (job 275685, 14-iter, kernel-filtered)**:

| metric                       | E28           | **E30 (1024)** | Δ |
|------------------------------|--------------:|---------------:|---:|
| Wavefront Occupancy          | 299.5/3328 (9.0 %) | **1654.6/3328 (49.7 %)** | **+5.5×** |
| IPC                          | 0.35          | **0.74**       | **+111 %** |
| IPC (Issued)                 | 0.83          | 0.79           | small drop |
| VALU Active Threads          | 29.3/64 (45.78 %) | 29.89/64 (46.7 %) | +2 % |
| VALU Utilization             | 10.42 %       | (similar)      | unchanged |
| LDS Bank Conflicts/Access    | 0.03          | 0.03           | unchanged |
| Active CUs                   | 82/104 (78.8 %) | **104/104 (100 %)** | +27 % |
| HBM Bandwidth                | 0 Gb/s        | 1638 Gb/s peak | now exercising HBM more, but L2 hit still high |
| VGPRs / SGPRs / scratch      | 116 / 112 / 0 | 116 / 112 / 0  | unchanged |

**Diagnosis**:

1. **Wavefront occupancy quintupled** (9 % → 49.7 %) — exactly because
   the same single block per CU now contains 4× more waves (16 vs 4).
   Same LDS, same per-wave VGPR pressure, just more waves packed in.
2. **IPC doubled** (0.35 → 0.74). With ~16 active waves per CU instead
   of 4, the scheduler has enough in-flight work to hide LDS latency
   between issue slots. This is the direct mechanism for the speedup.
3. **VALU active threads barely moved** (45.8 % → 46.7 %), confirming
   the per-wave divergence pattern is unchanged — the win is *not*
   from reducing divergence but from hiding it across more waves.
4. **Active CUs jumped to 100 %** (was 78.8 %) — with smaller blocks
   the early/last CU launch latency mattered; with fewer, larger
   blocks the scheduler now keeps every CU busy through the whole
   kernel duration.
5. HBM now shows nonzero traffic; that's because with 4× more compute
   throughput per CU, the per-zone LDS-staging at block entry and
   xout writeback at block end dominate the HBM time. Still small
   relative to total runtime.

**Decision**: **KEPT**. Largest single-experiment win since E28
(itself the largest since E19). Failure counter reset 1 → 0.

**Cumulative state at end of E30**:

| metric                | original baseline | **E30**     | Δ |
|-----------------------|------------------:|------------:|---:|
| avg_gpu_time (104z, 2it) | 46.7017 s     | **6.428 s** | **−86.2 %** |
| VGPRs                 | 117              | 116        | −1 |
| Wavefront occupancy   | 4 waves/CU       | **16 waves/CU (49.7 % of peak)** | +4× |
| IPC                   | 0.10 (baseline) | 0.74        | +7× |
| VALU active threads   | 21.6 %           | 46.7 %     | +116 % |
| Active CUs            | (n/a, smaller blocks) | 100 % | maxed |

**Observations for next phase**:

- The cheap launch-config win has been spent (1024 is the hardware
  max). Further wins must come from kernel-code changes again.
- IPC at 0.74 / 5.0 = 14.8 % of peak still leaves plenty of headroom.
- VALU active threads still only 47 % — divergence remains the
  long-term ceiling. Now that latency hiding is sufficient, the
  bottleneck shifts back to per-wave critical-path divergence in the
  j-loop tails of small species. Promising next directions:
  1. Process **multiple small species per wave** at sub-wave
     granularity (subwave = 16 lanes / species) so short j-loops use
     fewer lanes but still keep the wave fully active.
  2. Pad short j-ranges to a multiple of 16/32 with
     sentinel `(fac=0, map=SIZE)` entries (uses the existing
     `xout_zone[SIZE] = 1.0` trick) to eliminate tail divergence
     within the per-species reduction.
  3. Look at the rate-eval loop and energy/norm reductions, which
     are now a larger relative fraction of total time.


### Experiment 31 — Sub-wave species packing: 32-lane subwaves (2 species per wave)

**Intent**: act on the E30 observation that VALU active threads is only
46.7 % — most species have j-count well under 64, so the 64-lane
reduction in E28/E30 leaves half the lanes idle on the single j-loop
iteration each species needs. Split each wave into two 32-lane
subwaves; each subwave updates a different species concurrently with
the other half of the wave. The j-loop strides by 32, the reduction
collapses 32 lanes (5 shfl), and the per-block species count drops
from ~10 species per wave (E30) to ceil(SIZE / (2*nw)) = 5 per
(wave, subwave).

**Change** (`src/hipcore/bn_burner_gpu.hip`, species-update block):

```c
const int subwave = lane >> 5;     // 0 or 1
const int sublane = lane & 31;     // 0..31
const int nsub    = nw << 1;
for (int i = wave * 2 + subwave; i < SIZE; i += nsub) {
    // ...
    for (int j = p0 + sublane; j <= p1; j += 32)
        plus += (double)fp_fac_lds[j] * flux[(int)fp_map_lds[j]];
    for (int offset = 16; offset > 0; offset >>= 1)
        plus += __shfl_down(plus, offset, 32);
    // ... same for minus, then sublane==0 does asym/Euler scalar ...
}
```

**Timing (104 zones, 2 iter, 2 runs)**:

| run | avg per-iter | cycles/zone-batch |
|-----|-------------:|------------------:|
| 275687 | 3.2640 s | 76 892 352 |
| 275688 | 3.2568 s | 76 722 122 |

**Average E31**: **3.260 s** vs E30 6.428 s → **−49.3 %** in this
experiment alone, **−93.0 % CUMULATIVE** vs original baseline 46.7 s.

**Numerics**: bit-identical to baseline.

**Decision**: **KEPT** — but immediately superseded by E32 below
(further sub-wave shrink to 16 lanes won another 7.5 %).


### Experiment 32 — Sub-wave species packing: 16-lane subwaves (4 species per wave) — **CURRENT**

**Intent**: continue the E31 sweep. If 32 lanes → 2 species per wave
won 49 %, does 16 lanes × 4 species per wave win more?

**Change** (`src/hipcore/bn_burner_gpu.hip`):

```c
const int subwave = lane >> 4;     // 0..3
const int sublane = lane & 15;     // 0..15
const int nsub    = nw << 2;       // 4 species per wave
for (int i = wave * 4 + subwave; i < SIZE; i += nsub) {
    // ...
    for (int j = p0 + sublane; j <= p1; j += 16)
        plus += (double)fp_fac_lds[j] * flux[(int)fp_map_lds[j]];
    for (int offset = 8; offset > 0; offset >>= 1)
        plus += __shfl_down(plus, offset, 16);
    // ...
}
```

**Timing (104 zones, 2 iter, 2 runs)**:

| run | avg per-iter | cycles/zone-batch |
|-----|-------------:|------------------:|
| 275690 | 3.0218 s | 71 187 623 |
| 275691 | 3.0228 s | 71 210 282 |
| 275696 (rebuild verify) | 3.0209 s | 71 166 363 |
| 275697 (rebuild verify) | 3.0230 s | 71 214 260 |

**Average E32**: **3.022 s** vs E31 3.260 s → **−7.3 %** in this
experiment alone. vs E30 6.428 s → **−53.0 %** since E30. vs original
baseline 46.7017 s → **−93.5 % CUMULATIVE**.

**Numerics**: bit-identical to baseline.

**Assembly (`asm_e32/e32.s`)**: 28 056 lines (vs E30 28 823, −2.7 %),
VGPR=119, SGPR=100, scratch=0. Smaller because the 16-wide reduction
emits 4 shfl_down ops instead of 6.

**rocprof-compute (job 275698, 14-iter, kernel-filtered)**:

| metric                       | E30 (1024)   | **E32 (16-lane)** | Δ |
|------------------------------|-------------:|------------------:|---:|
| VALU Active Threads          | 29.89 (46.7 %) | **35.75 (55.86 %)** | **+20 %** |
| IPC                          | 0.74         | 0.56              | −24 % |
| IPC (Issued)                 | 0.79         | 0.88              | +11 % |
| Wavefront Occupancy          | 49.72 %      | 49.69 %           | unchanged |
| Active CUs                   | 100 %        | 100 %             | unchanged |
| LDS Bank Conflicts/Access    | 0.03         | 0.14              | +0.11 (still tiny) |
| VGPRs / SGPRs / scratch      | 116 / 112 / 0 | 120 / 112 / 0    | VGPR +4 |

**Diagnosis**:

1. **VALU active threads jumped from 47 % to 56 %** — exactly what we
   were aiming for. The 16-lane reduction matches the typical species
   j-count distribution much better than 64-lane: most species need
   only 1 iteration of the lane-strided loop, and 16 lanes covering
   ~10–30 j-entries gives a much higher utilization ratio than 64
   lanes covering the same range.
2. **IPC dropped 0.74 → 0.56**. Counter-intuitive at first, but it's
   because the kernel does *fewer* total instructions per timestep
   (smaller reductions, smaller per-species overhead): IPC = inst /
   cycle, the cycle count dropped much faster than the instruction
   count, so the *ratio* fell. IPC (Issued) actually *rose* to 0.88,
   meaning the scheduler is issuing nearly an instruction per cycle
   per wave on average — closer to a fundamental ceiling.
3. **VGPR went 116 → 120**. The four-way subwave indexing
   (`subwave = lane >> 4`, `sublane = lane & 15`) adds a couple of
   live integer values across the j-loop. Tiny cost, no occupancy
   impact (still LDS-bound at 1 block / CU).
4. **LDS bank conflicts up 0.03 → 0.14**. Stride-1 reads of
   `fp_fac_lds[j]` and `fp_map_lds[j]` at `j = p0 + sublane` (sublane
   in [0, 16)) hit a 16-way pattern over 32 banks — average ~2-way
   conflict. Contributes ~10s of cycles per timestep, well below the
   ~1 ms per-zone savings.
5. Wave occupancy and Active CUs are unchanged (still LDS-bound at
   1 block / CU, all 104 CUs busy).

**Decision**: **KEPT**. Largest single-experiment win since E30
(itself the largest since E28). Failure counter still 0.

**Cumulative state at end of E32**:

| metric                | original baseline | **E32**     | Δ |
|-----------------------|------------------:|------------:|---:|
| avg_gpu_time (104z, 2it) | 46.7017 s     | **3.022 s** | **−93.5 %** (15.5× speedup) |
| VGPRs                 | 117              | 120        | +3 |
| Wavefront occupancy   | 4 waves/CU       | 16 waves/CU | +4× |
| IPC (Issued)          | low              | 0.88       | near peak |
| VALU active threads   | 21.6 %           | 55.9 %     | +159 % |
| Active CUs            | partial          | 100 %      | maxed |


### Experiment 33 — 8-lane subwaves (8 species per wave) — **REVERTED**

**Intent**: continue the E31→E32 sub-wave shrink to see if 8-lane
subwaves win further.

**Change**: same shape as E32 but `subwave = lane >> 3`,
`sublane = lane & 7`, j-stride 8, 3-shfl reduction over 8 lanes,
8 species per wave.

**Timing (104 zones, 2 iter, 2 runs)**:

| run | avg per-iter | cycles/zone-batch |
|-----|-------------:|------------------:|
| 275693 | 4.2840 s | 100 922 559 |
| 275694 | 4.2798 s | 100 821 852 |

**Average E33**: **4.282 s** vs E32 3.022 s → **+41.7 %
REGRESSION**.

**Numerics**: bit-identical (correctness preserved).

**Why**: 8 lanes per species is now narrower than the typical
j-count (~10–30 for this network), so most species need 2–4
iterations of the j-loop instead of 1. Each extra iteration costs
the wave a full ~6-cycle issue regardless of how few lanes are
active in it, AND the per-species overhead (loop bookkeeping, the
3-shfl reduction, the lane==0 scalar update) is now amortized over
fewer j-iterations. Net: more iterations × more per-species
overhead trumps the marginal lane-utilization gain.

**Decision**: **REVERTED** to E32 (16-lane). Failure counter 0 → 1.

**Sweep summary** (species update sub-wave width, blockDim=1024,
104 zones, 2 iter):

| sub-wave width | species/wave | avg time | cycles | Δ vs prev |
|---------------:|-------------:|---------:|-------:|---------:|
| 64 (E30)       | 1 (full wave)| 6.428 s  | 151.4M | — |
| 32 (E31)       | 2            | 3.260 s  |  76.8M | −49.3 % |
| **16 (E32)**   | **4**        | **3.022 s** | **71.2M** | **−7.3 %** |
| 8  (E33)       | 8            | 4.282 s  | 100.9M | +41.7 % |

Sweet spot is **16 lanes / species** — matches the typical species
j-count for this network (most species ~10–30 contributing fluxes).

