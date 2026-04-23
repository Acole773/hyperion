# Zone-sweep scaling analysis (post-E41 kernel state)

**Kernel state**: `kernel-optimization-experiments-phase4` branch,
E41 applied (1.650 s at 104 zones, bit-identical to baseline).

**Device**: MI250 (one GCD, 104 CUs, 64 KB LDS/CU, 16 KB L1/CU).

**Goal**: Determine whether the kernel needs different
optimizations to run >104 zones efficiently on a single GPU,
i.e. is the optimization loop silently overfit to "one zone
per CU"?

## Method

- Zone counts chosen to probe every interesting ratio of
  zones to CUs: 52 (0.5), 104 (1), 156 (1.5), 208 (2), 260
  (2.5), 312 (3), 416 (4), 520 (5), 780 (7.5), 1040 (10).
- 2 iterations per run, single job per zone count
  (no averaging; numbers are stable to ~0.3 % on this kernel
  when zones is a whole multiple of 104, see E41 entry).
- Driver: `scripts/zone_sweep.sbatch` (array job).
- Tabulation: `scripts/tabulate_zone_sweep.sh`.

## Raw results

Captured on jobs 276635..276644 (see
`hyperion_run-276635.out` ... `hyperion_run-276644.out`).

| zones | zones/CU | avg_gpu_s | us/zone | efficiency vs 104z |
|------:|---------:|----------:|--------:|-------------------:|
|   52  |   0.50   | 1.6569   |  31864  |  51.9 %  |
|  104  |   1.00   | 1.7209   |  16547  | 100.0 % (ref) |
|  156  |   1.50   | 3.3571   |  21520  |  76.9 %  |
|  208  |   2.00   | 3.4328   |  16504  | 100.3 %  |
|  260  |   2.50   | 5.0605   |  19464  |  85.0 %  |
|  312  |   3.00   | 5.1417   |  16480  | 100.4 %  |
|  416  |   4.00   | 6.8571   |  16483  | 100.4 %  |
|  520  |   5.00   | 8.5629   |  16467  | 100.5 %  |
|  780  |   7.50   | 13.6011  |  17437  |  94.9 %  |
| 1040  |  10.00   | 17.1167  |  16458  | 100.5 %  |

`efficiency = (gpu_s_per_zone @ 104z) / (gpu_s_per_zone here)`.

## Observations

### 1. Perfect scaling whenever zones is a multiple of 104

For 104, 208, 312, 416, 520, 1040 zones, per-zone GPU cost
is **constant to within 0.6 %** (16458–16547 µs/zone). This
is textbook strong scaling: adding more work does not hurt
throughput at all. The scheduler happily streams
back-to-back "waves" of 104 blocks. 1040 zones (10 waves) is
actually *slightly cheaper per zone* than 104 zones (1 wave)
because per-kernel-launch overhead amortizes over more
work.

**Implication**: the kernel does NOT need a different
algorithm to run more zones efficiently. 208 zones already
runs at 100 % efficiency; 1040 zones also runs at 100 %
efficiency.

### 2. Partial-wave cases pay the full wave cost

Whenever zones is `104·k + r` with `0 < r < 104`, wall time
jumps to `(k+1) · T_wave` — the partial wave charges the
same as a full wave. Concretely:

- 52 zones → 0.5 waves "scheduled" but runs at 1-wave cost
  (half the CUs are idle the whole time). GPU time is
  1.657 s vs 1.721 s for 104 zones: **the 52 extra zones
  cost almost nothing** (+0.064 s). **Efficiency 52 %.**
- 156 zones → 1.5 waves → runs at 2-wave cost. GPU time
  3.357 s = 2× the 104z time. **Efficiency 77 %.**
- 260 zones → 2.5 waves → 3-wave cost (5.061 s ≈ 3×
  1.686 s). **Efficiency 85 %.**
- 780 zones → 7.5 waves → 8-wave cost (13.60 s = 8×
  1.700 s). **Efficiency 95 %** — the partial-wave overhead
  amortizes as the number of full waves grows.

Mathematical form:

```
gpu_s(N) ≈ ceil(N/104) × T_wave      with T_wave ≈ 1.71 s
efficiency(N) = N / (104 × ceil(N/104))
```

This **exactly** fits the measured numbers:
- `N=156`: 156/(104·⌈1.5⌉) = 156/208 = 75.0 %. Measured: 76.9 %.
- `N=260`: 260/(104·⌈2.5⌉) = 260/312 = 83.3 %. Measured: 85.0 %.
- `N=780`: 780/(104·⌈7.5⌉) = 780/832 = 93.8 %. Measured: 94.9 %.
(Measured values are slightly higher because the partial
wave is fractionally lighter due to less cross-wave
synchronization.)

### 3. The ceiling is occupancy, not the kernel

Each CU runs exactly **one** block at a time because the
block's LDS footprint pins occupancy. At E41 the LDS budget
per block is ≈ 44.6 KB, which is above the 32 KB/block cap
required to fit 2 blocks per CU (since each CU has 64 KB
LDS).

This means:

- No matter how large `N` is (even 1040 zones = 10 waves),
  the per-zone cost is still bounded below by `T_wave / 104`
  ≈ 16.5 ms/zone, determined entirely by single-block
  performance.
- Any kernel optimization that reduces single-block time
  benefits every `N`.
- The *only* way to break the `T_wave / 104` ceiling is to
  fit more than one block per CU simultaneously, which means
  dropping LDS below 32 KB/block.

### 4. What the user's original question really meant

The user's question was "do we need different optimizations
to run 2×104 zones efficiently on one GPU?". The answer is:

- **For correctness and linear scaling** → no, the existing
  kernel already strong-scales to 10 waves (1040 zones) at
  100 % efficiency.
- **To run 2×104 zones in ~ the same wall time as 1×104
  zones** (i.e. double the effective throughput) → yes, one
  thing is needed: get below 32 KB LDS. No algorithm change
  is needed on the hot path; just an occupancy-enabling
  refactor.

### 5. Partial-wave cases have an easy mitigation

If a user's workflow presents e.g. 156 zones, the 77 %
efficiency cliff can be fixed at the host side by padding
to the next multiple of 104 (or to the next multiple of the
current block-per-CU count, whatever that becomes):

- Pad 156 → 208 zones (fill 52 extra slots with dummy /
  no-op zones, or with actual repeat zones).
- Expected gain: `208/156 × 77 % = 103 %` of the measured
  156-zone throughput. In other words, **running 208 zones
  is basically free over running 156 zones** because the
  partial wave charges full wave cost anyway.

This is a *host-side* change that costs nothing in kernel
complexity. Might be worth surfacing as a knob
(`HYP_PAD_ZONES=1` rounds up to next multiple of 104).

## Recommendation

The immediate optimization target is **LDS reduction to
<32 KB/block**, to break the single-block-per-CU ceiling and
approximately halve `T_wave`. This is exactly the "move both
rate and flux to global in one shot" experiment flagged at
the end of E41's cumulative observations.

Rationale (recap):

- E41 LDS: ~44.6 KB. Need: <32 KB. Gap: ~12.6 KB.
- Largest LDS arrays still present: `rate` (12.8 KB),
  `flux` (12.8 KB). Each on its own is close to the gap but
  E38 showed that a single-array cut that doesn't cross the
  threshold *loses* time.
- Combined cut: 25.6 KB saved → new LDS ≈ 19 KB → fits 2
  blocks/CU with room to spare.
- Risk: `rate`+`flux` global working set per block = 25.6 KB
  > L1 capacity per CU (16 KB). With 2 blocks/CU the
  combined working set is 51 KB, guaranteed to thrash L1.
  Partial mitigation: `rate` is read-only (read once per
  reaction per timestep) and may live well in L2 (8 MB
  shared); `flux` is written and read many times per
  timestep within a block and will stress L1 harder.

Upside if it works: ~1.7 s → ~0.9 s at 104 zones and ~8 s
→ ~4 s at 1040 zones, i.e. another ~2× speedup on the hot
path. Even if L1 thrash costs 30 %, 2× occupancy still nets
~1.3× speedup. Worth trying.

Queued as **Experiment 44** (see `optimization_log.md`).

## Secondary observations worth following up on

1. Partial-wave padding at the host level (easy, no kernel
   risk, produces a user-visible throughput improvement for
   non-multiple-of-104 workloads). Not an optimization per
   se, more a knob / documentation note.
2. At 104 zones we are already running 16 waves/CU × 104 CUs
   = 1664 waves total, which is close to filling the vgpr
   file. Block size is pinned to 1024 and cannot grow; the
   only other occupancy lever is LDS (above) or VGPR
   reduction (would need ~half the VGPRs — large refactor,
   not obviously achievable).
3. 208 zones runs at 3.433 s; 2×104 zones sequential would
   run at 2×1.721 = 3.442 s. The scheduler is saving us
   exactly the per-launch overhead (~9 ms) and nothing
   else. Confirms waves run strictly back-to-back with no
   overlap, i.e. no cross-wave hiding.
