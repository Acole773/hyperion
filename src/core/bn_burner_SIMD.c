#include "bn_burner.h"
#include "store.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <threads.h>

#define __DIAG_ON
#include "diagnostic.h"

// #include <immintrin.h> // Theoretically this is portable
#include <x86intrin.h> // This is not

#define SIMD_WIDTH 8

#define ZERO_FIX 1e-50
#define THIRD 1.0 / 3.0

// So that my LSP stops bugging me.
#ifndef SIZE
#define SIZE 16
#endif

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
// #define SIZE 365
// #define NUM_REACTIONS 4395
// #define NUM_FLUXES_PLUS 7429
// #define NUM_FLUXES_MINUS 7420
#endif

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* restrict xout,
                                   double* sdotrate);

void hyperion_burner_(double* tstep, double* temp, double* dens, double* xin,
                      double* restrict xout, double* sdotrate,
                      uchar* burned_zone, int* size) {

    int rem = *size % SIMD_WIDTH;

    for (int i = 0; i < *size / SIMD_WIDTH; i++) {
        hyperion_burner_kernel(
            tstep, &temp[i * SIMD_WIDTH], &dens[i * SIMD_WIDTH],
            xin + (SIZE * i * SIMD_WIDTH), xout + (SIZE * i * SIMD_WIDTH),
            &sdotrate[i * SIMD_WIDTH]);
    }

    // TODO: handle rem (non simd width encompassed values, so we have to pass
    // nulls for values...) for (int i = 0; i < rem; i++) {
    // }
}

static void hyperion_burner_kernel(double* tstep, double* temp, double* dens,
                                   double* xin, double* restrict xout,
                                   double* sdotrate) {

    // TODO: just like in the serial code, we call abundance X (mass frac)
    // because it's what is passed to the func.
    __m512d _xout[SIZE];

    // TODO: all the vector registers here should be handled outside once,
    // outside of this function.
    __m512d _aa[SIZE];
    __m512d _f_plus_factor[NUM_FLUXES_PLUS];
    __m512d _f_minus_factor[NUM_FLUXES_MINUS];
    __m512d _q[NUM_REACTIONS];
    for (int i = 0; i < NUM_FLUXES_PLUS; i++) {
        _f_plus_factor[i] = _mm512_set1_pd(f_plus_factor[i]);
    }
    for (int i = 0; i < NUM_FLUXES_MINUS; i++) {
        _f_minus_factor[i] = _mm512_set1_pd(f_minus_factor[i]);
    }
    for (int i = 0; i < NUM_REACTIONS; i++) {
        _q[i] = _mm512_set1_pd(q_value[i]);
    }
    for (int i = 0; i < SIZE; i++) {
        _aa[i] = _mm512_set1_pd(aa[i]);
    }

    __m512d _temp = _mm512_loadu_pd(temp);
    _temp = _mm512_div_pd(_temp, _mm512_set1_pd(1e9));
    __m512d _dens = _mm512_loadu_pd(dens);
    __m512d _one = _mm512_set1_pd(1.0);
    __m512d _zero_fix = _mm512_set1_pd(ZERO_FIX);
    __m512d _econ = _mm512_set1_pd(
        9.5768e17); // Conversion thingy (MeV/nucleon/s to erg/g/s)

    // General simple vector registers
    __m512d _tmp;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIMD_WIDTH; j++) {
            _xout[i][j] = xin[i + (SIZE * j)];
        }
    }
    for (int i = 0; i < SIZE; i++) {
        _xout[i] = _mm512_div_pd(_xout[i], _aa[i]);
    }

    __m512d _ddt_e = _mm512_set1_pd(0);
    __m512d _t5;
    __m512d _t6;
    // _t3 is := _t93
    // _t4 is := _temp
    // Ugh. I need import some libraries and whatnot for this...
    // This is good enough for now though
    __m512d _t93;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        _t93[i] = cbrt(_temp[i]);
        _t5[i] = pow(_t93[i], 5);
        _t6[i] = log(_temp[i]);
    }
    // __m512d _t1 = _mm512_rcp14_pd(_temp); // Reciprocal with error 2^-14
    // __m512d _t2 = _mm512_rcp14_pd(_t93);
    __m512d _t1 = _mm512_div_pd(_one, _temp);
    __m512d _t2 = _mm512_div_pd(_one, _t93);

    __m512d _density[3] = {_mm512_set1_pd(1.0), _dens,
                           _mm512_mul_pd(_dens, _dens)};

    __m512d _rate[NUM_REACTIONS];
    for (int i = 0; i < NUM_REACTIONS; i++) {
        __m512d _prefactor = _mm512_mul_pd(_mm512_set1_pd(prefactor[i]),
                                           _density[num_react_species[i] - 1]);
        // TODO: again, we need better SIMD here.
        for (int j = 0; j < SIMD_WIDTH; j++) {
            _tmp[j] = exp(p_0[i] + _t1[j] * p_1[i] + _t2[j] * p_2[i] +
                          _t93[j] * p_3[i] + _temp[j] * p_4[i] +
                          _t5[j] * p_5[i] + _t6[j] * p_6[i]);
        }
        _rate[i] = _mm512_mul_pd(_prefactor, _tmp);
    }

    // NOTE: it is clear that for this SIMD idea to work, all of the cells must
    // be able to be evolved in the same timescale. This could requires some
    // specification (similar temperatures, whatever), but we could also just
    // give some cells _more_ temporal resolution than they need, since no
    // matter how we slice the problem, we are waiting on them anyway. (it's not
    // quite as efficient, since SIMD _is_ a little slower than normal ops, but
    // I'd guess it's still faster)...
    double t = 1e-20;
    double dt = t * 1e-2;

    // These size may or may not be entirely correct...
    __m512d _flux[NUM_REACTIONS];
    __m512d _f_plus_sum[SIZE];
    __m512d _f_minus_sum[SIZE];
    __m512d _f_plus[NUM_FLUXES_PLUS];
    __m512d _f_minus[NUM_FLUXES_MINUS];
    while (t < *tstep) {
        for (int i = 0; i < NUM_REACTIONS; i++) {
            switch (num_react_species[i]) {
            case 1:
                _flux[i] = _mm512_mul_pd(_rate[i], _xout[reactant_1[i]]);
                break;
            case 2:
                _flux[i] = _mm512_mul_pd(_rate[i], _xout[reactant_1[i]]);
                _flux[i] = _mm512_mul_pd(_flux[i], _xout[reactant_2[i]]);
                break;
            case 3:
                _flux[i] = _mm512_mul_pd(_rate[i], _xout[reactant_1[i]]);
                _flux[i] = _mm512_mul_pd(_flux[i], _xout[reactant_2[i]]);
                _flux[i] = _mm512_mul_pd(_flux[i], _xout[reactant_3[i]]);
                break;
            }
        }

        for (int i = 0; i < f_plus_total; i++) {
            _f_plus[i] = _mm512_mul_pd(_f_plus_factor[i], _flux[f_plus_map[i]]);
        }
        for (int i = 0; i < f_minus_total; i++) {
            _f_minus[i] =
                _mm512_mul_pd(_f_minus_factor[i], _flux[f_minus_map[i]]);
        }

        for (int i = 0; i < SIZE; i++) {
            // f_*_max uses a leading sentinel ([0] = -1); no (i == 0) guard.
            int min_plus = f_plus_max[i] + 1;
            int min_minus = f_minus_max[i] + 1;
            _f_plus_sum[i] = _mm512_setzero_pd();
            for (int j = min_plus; j <= f_plus_max[i + 1]; j++) {
                _f_plus_sum[i] = _mm512_add_pd(_f_plus_sum[i], _f_plus[j]);
            }
            _f_minus_sum[i] = _mm512_setzero_pd();
            for (int j = min_minus; j <= f_minus_max[i + 1]; j++) {
                _f_minus_sum[i] = _mm512_add_pd(_f_minus_sum[i], _f_minus[j]);
            }
        }

        __m512d _dt = _mm512_set1_pd(dt);

        for (int i = 0; i < SIZE; i++) {
            _tmp = _mm512_mul_pd(_f_minus_sum[i], _dt);

            __mmask8 _kmask = _mm512_cmp_pd_mask(_tmp, _xout[i],
                                                 _CMP_GT_OQ); // TODO: OQ vs OS

            // ASY
            __m512d _tmpp;
            _tmp = _mm512_mul_pd(_dt, _f_plus_sum[i]);
            _tmpp = _xout[i];
            _xout[i] = _mm512_mask_add_pd(_xout[i], _kmask, _tmp, _xout[i]);

            _tmp = _mm512_add_pd(_tmpp, _zero_fix);
            _tmp = _mm512_div_pd(_f_minus_sum[i], _tmp);

            _tmp = _mm512_fmadd_pd(_tmp, _dt, _one);
            _xout[i] = _mm512_mask_div_pd(_xout[i], _kmask, _xout[i], _tmp);

            // FE
            _kmask = _knot_mask8(_kmask);
            _tmp = _mm512_sub_pd(_f_plus_sum[i], _f_minus_sum[i]);
            _tmp = _mm512_mul_pd(_dt, _tmp);
            _xout[i] = _mm512_mask_add_pd(_xout[i], _kmask, _tmp, _xout[i]);
        }

        __m512d _normalfac = _mm512_setzero_pd();
        for (int i = 0; i < SIZE; i++) {
            _xout[i] = _mm512_mul_pd(_aa[i], _xout[i]);
            _normalfac = _mm512_add_pd(_normalfac, _xout[i]);
        }
        _normalfac = _mm512_div_pd(_one, _normalfac);
        // _normalfac = _mm512_rcp14_pd(_normalfac);
        for (int i = 0; i < SIZE; i++) {
            _xout[i] = _mm512_mul_pd(_xout[i], _normalfac);
            _xout[i] = _mm512_div_pd(_xout[i], _aa[i]);
        }

        _tmp = _mm512_setzero_pd();
        for (int i = 0; i < num_reactions; i++) {
            _tmp = _mm512_fmadd_pd(_flux[i], _q[i], _tmp);
        }
        _tmp = _mm512_mul_pd(_tmp, _dt);
        _ddt_e = _mm512_fmadd_pd(_tmp, _econ, _ddt_e);

        t += dt;
        dt = 1e-2 * t;
    }

    // This is my greatest sin.
    for (int i = 0; i < SIZE; i++) {
        _xout[i] = _mm512_mul_pd(_aa[i], _xout[i]);
        for (int j = 0; j < SIMD_WIDTH; j++) {
            xout[i + (j * SIZE)] = _xout[i][j];
        }
    }
    _mm512_store_pd(sdotrate, _ddt_e);

    return;
}
