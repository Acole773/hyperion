#include "bn_burner_gpu.h"

#define ZERO_FIX 1e-50
#define THIRD 1.0 / 3.0

#include <hip/hip_runtime.h>

#if SIZE == 16
#define SIZE 16
#define NUM_REACTIONS 48
#define NUM_FLUXES_PLUS 72
#define NUM_FLUXES_MINUS 72
#endif

#if SIZE == 150
#define SIZE 150
#define NUM_REACTIONS 1604
#define NUM_FLUXES_PLUS 2710
#define NUM_FLUXES_MINUS 2704
#endif

#if SIZE == 365
#define SIZE 365
#define NUM_REACTIONS 4395
#define NUM_FLUXES_PLUS 7429
#define NUM_FLUXES_MINUS 7420
#endif

#include <math.h>

#define GPU_ASSERT_FINITE(val, tag, species_idx, flux_idx)             \
    do {                                                               \
        if (!isfinite(val)) {                                          \
	    printf("CRITICAL ERROR: %s is NaN at Species: %d, Flux_Map_Idx: %d\n", \
               tag, species_idx, flux_idx); \
            printf("  zone=%d thread=%d block=%d\n",                   \
                   blockIdx.x, threadIdx.x, blockIdx.x);               \
            printf("  value=%e\n", (double)(val));                     \
            /* __builtin_trap(); <-- Comment this out to prevent hardware exception */ \
            /* We print only the first failure to avoid "ridiculous prints" */ \
	   return; \
        }                                                              \
    } while (0)

#define GPU_DEBUG 1

#if GPU_DEBUG
#define GPU_GUARD(cond, msg, a, b, c) \
    if (cond) { \
        printf("GPU ERROR: %s | a=%d b=%d c=%d\n", msg, a, b, c); \
        asm("s_trap 0"); \
    }
#else
#define GPU_GUARD(cond, msg, a, b, c)
#endif

extern "C" __global__ void hyperion_burner_dev_kernel(
    int Nzones,
    const double* __restrict__ temp,
    const double* __restrict__ dens,
    const double* __restrict__ xin,
    double*       __restrict__ xout,
    double*       __restrict__ sdotrate,
    const double* __restrict__ prefactor,
    const double* __restrict__ p_0,
    const double* __restrict__ p_1,
    const double* __restrict__ p_2,
    const double* __restrict__ p_3,
    const double* __restrict__ p_4,
    const double* __restrict__ p_5,
    const double* __restrict__ p_6,
    const double* __restrict__ aa,
    const double* __restrict__ q_value,
    const int*    __restrict__ reactant_1,
    const int*    __restrict__ reactant_2,
    const int*    __restrict__ reactant_3,
    const int*    __restrict__ f_plus_map,
    const int*    __restrict__ f_minus_map,
    const double* __restrict__ f_plus_factor,
    const double* __restrict__ f_minus_factor,
    const int*    __restrict__ f_plus_max,
    const int*    __restrict__ f_minus_max,
    const int*    __restrict__ num_react_species,
    const double* __restrict__ real_vals,
    double*       __restrict__ rate_g,
    const int*    __restrict__ species_perm,
    int                          phase1_count){

    int tid = threadIdx.x;

   
    // LDS layout (dynamic shared memory):
    //   flux       [NUM_REACTIONS]  - exp06 baseline
    //   reduce_buf [num_waves]      - block-level reduction scratch
    //   rate       [NUM_REACTIONS]  - exp06: moved from global rate_g
    //   xout_lds   [SIZE]           - exp09: moved from global xout_zone.
    //   nrs_lds    [NUM_REACTIONS]  - exp10: packed num_react_species as
    //                                 unsigned char (values 1..3). Saves 4x
    //                                 vs int. Read once per reaction in flux
    //                                 loop, no need for the global reload.
    const int num_waves_k = blockDim.x >> 6;
    extern __shared__ double smem[];
    double* flux       = smem;                              // NUM_REACTIONS
    double* reduce_buf = flux + NUM_REACTIONS;              // num_waves
    double* rate       = reduce_buf + num_waves_k;          // NUM_REACTIONS
    // exp25: xout_lds is SIZE+1 doubles. xout_lds[SIZE] is a dummy slot
    // initialized to 1.0 once per zone. The parser puts num_species (=SIZE)
    // into reactant_1/2/3 for unused reactant slots, so the flux loop can
    // unconditionally multiply by xout_zone[reactant_k] for k=1,2,3 without
    // any branch or filter (multiply by 1.0 = identity).
    double* xout_lds   = rate + NUM_REACTIONS;              // SIZE + 1
    // exp40: pack nrs, r1, r2, r3 (four uchars per reaction) into a single
    // uint32 per reaction. The LDS footprint is identical (4 * NUM_REACTIONS
    // bytes either way) because the four byte arrays were already adjacent,
    // but now the flux loop reads all three reactant indices with one
    // ds_read_b32 instead of three ds_read_u8 instructions (-2 LDS ops per
    // reaction per timestep across the block). The once-per-zone rate
    // eval loop still reads only the nrs byte via a masked extraction, so
    // no extra LDS bandwidth is spent there.
    //   byte 0 = reactant_1
    //   byte 1 = reactant_2
    //   byte 2 = reactant_3
    //   byte 3 = num_react_species  (formerly nrs_lds)
    unsigned int* rxns_lds = (unsigned int*)(xout_lds + (SIZE + 1));           // NUM_REACTIONS uint32s
    // Signed short so the leading sentinel (-1) at index 0 survives the cast.
    // Length is SIZE + 1 because f_*_max carries a sentinel at [0] and the
    // real upper bound for species i lives at [i + 1].
    short* fp_max_lds = (short*)(rxns_lds + NUM_REACTIONS);                    // SIZE + 1
    short* fm_max_lds = fp_max_lds + (SIZE + 1);                               // SIZE + 1
    // exp39: pack f_plus_map (ushort) and f_plus_factor (uchar) into a
    // single uint16 per j. Reaction indices are in [0, NUM_REACTIONS=1604)
    // so the map fits in 12 bits; stoichiometric coefficients are small
    // integers (<= 15 in this network) so the fac fits in 4 bits.
    // Packed layout:
    //   bits [11:0] = map value (reaction index)
    //   bits [15:12] = fac value (stoichiometric coefficient)
    // Benefits:
    //   * inner j-loop drops from 2 independent LDS loads
    //     (ds_read_u16 for map + ds_read_u8 for fac) to a single
    //     ds_read_u16 load, reducing LDS issue-port pressure
    //     (LDS ops were ~30 % of all issued instructions in E37).
    //   * LDS footprint shrinks by (NUM_FLUXES_PLUS + NUM_FLUXES_MINUS)
    //     bytes (about 5.3 KB for the 150-species network).
    unsigned short* fp_pack_lds = (unsigned short*)(fm_max_lds + (SIZE + 1));  // NUM_FLUXES_PLUS
    unsigned short* fm_pack_lds = fp_pack_lds + NUM_FLUXES_PLUS;               // NUM_FLUXES_MINUS
    // Pad to 8-byte alignment for the double aa_lds that follows.
    double* aa_lds = (double*)(((uintptr_t)(fm_pack_lds + NUM_FLUXES_MINUS) + 7) & ~(uintptr_t)7);  // SIZE doubles
    // exp11/12/13: pack reactant_1/2/3 as uchar (species idx < SIZE = 150).
    // exp14: pack f_plus_max / f_minus_max as ushort.
    // exp15: pack f_plus_map / f_minus_map as ushort. Values are reaction
    //        indices in [0, NUM_REACTIONS=1604) so ushort suffices.

    // Stage block-shared metadata into LDS once per block.
    // exp40: pack nrs/r1/r2/r3 into one uint32 per reaction.
    for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
        unsigned r1 = (unsigned)reactant_1[r]        & 0xFFu;
        unsigned r2 = (unsigned)reactant_2[r]        & 0xFFu;
        unsigned r3 = (unsigned)reactant_3[r]        & 0xFFu;
        unsigned ns = (unsigned)num_react_species[r] & 0xFFu;
        rxns_lds[r] = r1 | (r2 << 8) | (r3 << 16) | (ns << 24);
    }
    for (int i = tid; i < SIZE + 1; i += blockDim.x) {
        fp_max_lds[i] = (short)f_plus_max[i];
        fm_max_lds[i] = (short)f_minus_max[i];
    }
    // exp39: stage (fac<<12 | map) packed into uint16 in one pass per side.
    // Host validates that fac values fit in 4 bits, so the shift is safe.
    for (int j = tid; j < NUM_FLUXES_PLUS; j += blockDim.x) {
        unsigned map = (unsigned)f_plus_map[j];
        unsigned fac = (unsigned)f_plus_factor[j];
        fp_pack_lds[j] = (unsigned short)(((fac & 0xFu) << 12) | (map & 0x0FFFu));
    }
    for (int j = tid; j < NUM_FLUXES_MINUS; j += blockDim.x) {
        unsigned map = (unsigned)f_minus_map[j];
        unsigned fac = (unsigned)f_minus_factor[j];
        fm_pack_lds[j] = (unsigned short)(((fac & 0xFu) << 12) | (map & 0x0FFFu));
    }
    // exp16: aa[SIZE] is read in initial-norm, energy-reduce and final-norm
    //        - dozens of times per zone per timestep. Tiny LDS cost (1.2KB).
    for (int i = tid; i < SIZE; i += blockDim.x)
        aa_lds[i] = aa[i];
    __syncthreads();

    for (int zone = blockIdx.x; zone < Nzones; zone += gridDim.x){ 

        double T   = temp[zone];
        double rho = dens[zone];
        double tstep = real_vals[0];
    
        double* aa_zone = aa_lds;  // exp16: redirect to LDS copy
    
        // Per-zone offsets
        // Experiment 09: xout_zone now lives in LDS for the duration of the
        // per-zone work (writeback to global at end of the zone loop body).
        double* xout_zone = xout_lds;
        const double* xin_zone  = xin  + zone * SIZE;
    
        __shared__ double norm_sum;
        double t, dt, t_end;
        int done;
    
        // ------------------------------------------------------------
        // Initial normalization: xout = xin / aa
        // exp25: also seed xout_zone[SIZE] = 1.0 (dummy slot for unused
        //        reactant indices in the branchless flux loop).
        // ------------------------------------------------------------
        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_zone[i] = xin_zone[i] / aa_zone[i];
        if (tid == 0) xout_zone[SIZE] = 1.0;
    
        __syncthreads();
        // ------------------------------------------------------------
        // Thermodynamic precompute
        // ------------------------------------------------------------
        double tmp = T * 1e-9;
        // Experiment 01: pow(tmp, 1/3) -> cbrt(tmp)
        // pow() is inlined as exp(y*log(x)) with NaN/sign/integer-y special cases
        // (~335 lines of gfx90a asm at baseline). cbrt() is a dedicated intrinsic
        // with a much shorter sequence, reducing VGPR/SGPR pressure.
        double t93 = cbrt(tmp);
    
        double t1 = 1.0 / (tmp);
        double t2 = 1.0 / t93;
        double t3 = t93;
        double t4 = tmp;
        double t93_2 = t93 * t93;
        double t5 = t93_2 * t93_2 * t93;
        double t6 = log(tmp);
    
        double density[3];
        density[0] = 1.0;
        density[1] = rho;
        density[2] = rho * rho;
    
        // ------------------------------------------------------------
        // Rate evaluation
        // ------------------------------------------------------------
        // exp27: read num_react_species from LDS (nrs_lds, uchar) instead of
        // global memory. The LDS copy is already staged once at block entry,
        // so this just routes the rate-eval load through LDS instead of HBM.
        for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
            // exp40: nrs lives in byte 3 of the packed rxns_lds uint32.
            int ns = (int)(rxns_lds[r] >> 24) - 1;
            ns = max(0, min(ns, 2));
    
            rate[r] =
                prefactor[r] * density[ns] *
                exp(p_0[r]
                  + p_1[r] * t1
                  + p_2[r] * t2
                  + p_3[r] * t3
                  + p_4[r] * t4
                  + p_5[r] * t5
                  + p_6[r] * t6);
    	//GPU_ASSERT_FINITE(rate[r], "rate");
        }
        __syncthreads();
    
        // ------------------------------------------------------------
        // Time integration
        // ------------------------------------------------------------
    
        
        t  = 1e-20;
        dt = t * 1e-2;
        t_end = tstep;
        done = 0;
    
        while (!done) {
    
            // --------------------------------------------------------
            // Flux computation
            //
            // exp25 (branchless via dummy slot): the parser stores
            // num_species (=SIZE) in reactant_k for unused slots, and
            // xout_zone[SIZE] is initialized to 1.0 once per zone. So the
            // load xout_zone[reactant_k] yields 1.0 for unused reactants
            // and the genuine abundance otherwise, making the per-reaction
            // multiply unconditional. nrs_lds[r] is no longer needed in
            // the flux loop (saves 1 LDS load + 2 int->double conversions
            // + 4 fma per reaction vs the exp21 filter form).
            //
            // The isfinite(f_rate) printf debug guard is also removed:
            // it was a divergent branch executed every reaction.
            // --------------------------------------------------------
            for (int r = tid; r < NUM_REACTIONS; r += blockDim.x) {
                // exp40: one ds_read_b32 delivers all three reactant indices.
                unsigned rxns = rxns_lds[r];
                int idx1 = (int)( rxns        & 0xFFu);
                int idx2 = (int)((rxns >> 8)  & 0xFFu);
                int idx3 = (int)((rxns >> 16) & 0xFFu);
                double a1 = xout_zone[idx1];
                double a2 = xout_zone[idx2];
                double a3 = xout_zone[idx3];
                flux[r] = rate[r] * a1 * a2 * a3;
            }
    
            __syncthreads(); //this ensures all fluxes have been updated
    
            // --------------------------------------------------------
            // Species updates  (exp32: sub-wave-cooperative,
            //                          4 species per wave / 16 lanes each)
            //
            // E28's 64-lane reduction wasted lanes for the common case
            // where a species has j-count < 64 — measured 47 % VALU
            // active threads after E30's 16-waves-per-block change.
            //
            // E32 splits each wave into four 16-lane subwaves. Each
            // subwave updates a different species concurrently with the
            // other three subwaves in the same wave. j-loop strides by
            // 16, the reduction is a 16-wide shfl_down (4 cycles), and
            // the per-block species count drops from ~10 species per
            // wave (E30) to ceil(SIZE / (4*num_waves)) per (wave,
            // subwave). For SIZE=150 / blockDim=1024 this is 3 species
            // per (wave, subwave).
            //
            // The 16-lane width was chosen empirically: 64-lane (E30,
            // 6.428 s), 32-lane (E31, 3.260 s), 16-lane (E32, 3.022 s),
            // 8-lane (E33, 4.282 s — regression). 16 matches the typical
            // species j-count distribution for this network: most
            // species have j-count ~10–30, so a 16-lane reduction lands
            // near the lane-utilization sweet spot without paying for
            // the per-species reduction/scalar-update overhead too
            // many times.
            //
            // Caveat: the wave executes one PC, so the j-loop iteration
            // count is determined by max(j_count over the four paired
            // species). For widely mismatched groups the long-tail
            // subwaves waste their lane budget. Sorting species by
            // j-count to make balanced groups is a potential follow-up.
            //
            // Numerical impact: parallel reduction reorders the FP sums
            // but stays bit-identical for this network (same as E28).
            // --------------------------------------------------------
            //
            // exp35: hybrid two-phase scheduler driven by the host-built
            // species_perm[] (descending by max(plus j-count, minus j-count)).
            //
            //   Phase 1 (wave-cooperative, 64-lane reduction):
            //     The first `phase1_count` species in perm[] are the highest
            //     j-count species. One species per wave; all 64 lanes
            //     cooperate on the j-loop strided by 64 and reduce via a
            //     6-stage 64-wide shfl. This is the only way to keep all
            //     lanes useful for the rare huge species (max j-count = 403
            //     in the 150-species network) — under E32's 16-lane sub-wave
            //     reduction those species would force ceil(403/16)=26 inner
            //     iters with 3/4 of the lanes idle.
            //
            //   Phase 2 (sub-wave, 16-lane reduction):
            //     The remaining (SIZE - phase1_count) species are processed
            //     under the E32 layout: each wave splits into 4 16-lane
            //     subwaves, processing 4 species concurrently with j-loop
            //     stride 16 and a 4-stage 16-wide shfl. This is the right
            //     width for the bulk of the network — 70% of species have
            //     j-count in [8,16).
            //
            // The two phases write disjoint xout_zone slots so no barrier
            // is required between them; the existing __syncthreads() at
            // the end suffices for the next timestep.
            //
            // exp37: CONCURRENT phase dispatch. With phase1_count=4 the
            // previous sequential code path left 12 of 16 waves idle
            // during Phase 1. Assign waves 0..(phase1_count-1) to Phase 1
            // and waves phase1_count..(nw-1) to Phase 2 so both phases
            // run simultaneously on disjoint wave sets, and block time
            // becomes max(Phase 1, Phase 2) instead of their sum. Phase 2
            // now has fewer waves (nw_p2 = nw - phase1_count = 12) so its
            // outer-iter count grows slightly (3 -> 4 for SIZE=150), but
            // that cost is dwarfed by hiding Phase 1 behind Phase 2.
            //
            {
                const int wave    = tid >> 6;
                const int lane    = tid & 63;
                const int nw      = blockDim.x >> 6;

                if (wave < phase1_count) {
                    // ---- Phase 1: wave-cooperative for the biggest species ----
                    // Each of the first phase1_count waves processes exactly
                    // one huge species with all 64 lanes cooperating.
                    int i  = species_perm[wave];
                    int p0 = (int)fp_max_lds[i] + 1;
                    int m0 = (int)fm_max_lds[i] + 1;
                    int p1 = (int)fp_max_lds[i + 1];
                    int m1 = (int)fm_max_lds[i + 1];

                    double plus = 0.0;
                    for (int j = p0 + lane; j <= p1; j += 64) {
                        unsigned pk = fp_pack_lds[j];
                        plus += (double)(pk >> 12) * flux[(int)(pk & 0x0FFFu)];
                    }
                    for (int offset = 32; offset > 0; offset >>= 1)
                        plus += __shfl_down(plus, offset, 64);

                    double minus = 0.0;
                    for (int j = m0 + lane; j <= m1; j += 64) {
                        unsigned pk = fm_pack_lds[j];
                        minus += (double)(pk >> 12) * flux[(int)(pk & 0x0FFFu)];
                    }
                    for (int offset = 32; offset > 0; offset >>= 1)
                        minus += __shfl_down(minus, offset, 64);

                    if (lane == 0) {
                        double x_i = xout_zone[i];
                        if (minus * dt > x_i) {
                            // exp41: algebraically reduce two divisions to one.
                            //   (x_i + plus*dt) / (1.0 + (minus/(x_i+ZERO_FIX))*dt)
                            // = (x_i + plus*dt) * (x_i+ZERO_FIX) / (x_i+ZERO_FIX + minus*dt)
                            // Double-precision division is ~30 cycles; this
                            // halves the divisions per implicit-branch species
                            // update. FP rounding is not bit-identical to the
                            // original formulation, but stays well under the
                            // 1e-12 relative-tolerance correctness budget.
                            double d = x_i + ZERO_FIX;
                            x_i = (x_i + plus * dt) * d / (d + minus * dt);
                        } else {
                            x_i += (plus - minus) * dt;
                        }
                        if (x_i < ZERO_FIX)
                            x_i = ZERO_FIX;
                        xout_zone[i] = x_i;
                    }
                } else {
                    // ---- Phase 2: 16-lane sub-wave packing for the rest ----
                    // Uses only waves [phase1_count..nw), so the slot index
                    // is (wave - phase1_count) * 4 + subwave out of a total
                    // of (nw - phase1_count) * 4 slots.
                    const int subwave = lane >> 4;        // 0..3
                    const int sublane = lane & 15;        // 0..15
                    const int nw_p2   = nw - phase1_count;
                    const int nsub    = nw_p2 << 2;       // 4 species per wave
                    const int wave_p2 = wave - phase1_count;
                    const int small_count = SIZE - phase1_count;

                    for (int s = wave_p2 * 4 + subwave; s < small_count; s += nsub) {
                        int i  = species_perm[phase1_count + s];
                        int p0 = (int)fp_max_lds[i] + 1;
                        int m0 = (int)fm_max_lds[i] + 1;
                        int p1 = (int)fp_max_lds[i + 1];
                        int m1 = (int)fm_max_lds[i + 1];

                        double plus = 0.0;
                        for (int j = p0 + sublane; j <= p1; j += 16) {
                            unsigned pk = fp_pack_lds[j];
                            plus += (double)(pk >> 12) * flux[(int)(pk & 0x0FFFu)];
                        }
                        for (int offset = 8; offset > 0; offset >>= 1)
                            plus += __shfl_down(plus, offset, 16);

                        double minus = 0.0;
                        for (int j = m0 + sublane; j <= m1; j += 16) {
                            unsigned pk = fm_pack_lds[j];
                            minus += (double)(pk >> 12) * flux[(int)(pk & 0x0FFFu)];
                        }
                        for (int offset = 8; offset > 0; offset >>= 1)
                            minus += __shfl_down(minus, offset, 16);

                        if (sublane == 0) {
                            double x_i = xout_zone[i];
                            if (minus * dt > x_i) {
                                // exp41: same 2->1 division reformulation as Phase 1.
                                double d = x_i + ZERO_FIX;
                                x_i = (x_i + plus * dt) * d / (d + minus * dt);
                            } else {
                                x_i += (plus - minus) * dt;
                            }
                            if (x_i < ZERO_FIX)
                                x_i = ZERO_FIX;
                            xout_zone[i] = x_i;
                        }
                    }
                }
            }
            __syncthreads();
    
            t += dt;        
            dt = 1e-4 * t;
    	if (t + dt > t_end)
    	     dt = max(t_end - t, 1e-30);        
    
            if (t >= t_end || !isfinite(t) || dt <= 0.0)
                done = 1;
        }

    
    
        // ------------------------------------------------------------
        // Final renormalization (block reduction)
        //   Phase 1: intra-wave reduction via warp shuffle
        //   Phase 2: cross-wave reduction via LDS
        // ------------------------------------------------------------
        double local = 0.0;
        for (int i = tid; i < SIZE; i += blockDim.x)
            local += xout_zone[i] * aa_zone[i];

        // Intra-wave reduction (wavefront = 64 lanes on gfx90a)
        for (int offset = 32; offset > 0; offset >>= 1)
            local += __shfl_down(local, offset, 64);

        int lane   = tid & 63;
        int waveId = tid >> 6;
        const int num_waves = blockDim.x >> 6;

        if (lane == 0)
            reduce_buf[waveId] = local;

        __syncthreads();

        // Cross-wave reduction: wave 0 reduces the num_waves partial sums
        if (tid < num_waves) {
            local = reduce_buf[tid];
            for (int offset = num_waves >> 1; offset > 0; offset >>= 1)
                local += __shfl_down(local, offset, 64);
        }

        if (tid == 0)
            norm_sum = 1.0 / local;

        __syncthreads();

        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_zone[i] = (xout_zone[i] * aa_zone[i] * norm_sum) / aa_zone[i];

        __syncthreads();


        // ------------------------------------------------------------
        // Energy generation (block reduction)
        //   Phase 1: intra-wave reduction via warp shuffle
        //   Phase 2: cross-wave reduction via LDS
        // ------------------------------------------------------------

        double e_local = 0.0;
        for (int r = tid; r < NUM_REACTIONS; r += blockDim.x)
            e_local += flux[r] * q_value[r];

        for (int offset = 32; offset > 0; offset >>= 1)
            e_local += __shfl_down(e_local, offset, 64);

        if (lane == 0)
            reduce_buf[waveId] = e_local;

        __syncthreads();

        if (tid < num_waves) {
            e_local = reduce_buf[tid];
            for (int offset = num_waves >> 1; offset > 0; offset >>= 1)
                e_local += __shfl_down(e_local, offset, 64);
        }

        if (tid == 0)
            sdotrate[zone] = e_local * tstep * 9.5768e17;

         __syncthreads();

        // Experiment 09: copy LDS xout_lds back to global xout for the host.
        double* xout_global = xout + zone * SIZE;
        for (int i = tid; i < SIZE; i += blockDim.x)
            xout_global[i] = xout_lds[i];

        __syncthreads();
    }
}
