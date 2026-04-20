#ifndef __STORE_H
#define __STORE_H

#include <limits.h>

#ifndef PATH_MAX
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include "restrict.h"

extern int num_reactions;
extern int num_species;

extern char** reaction_label;
extern char** iso_label;

extern int* HYP_RESTRICT rg_member_idx;
extern int* rg_class;
extern int* reaction_library_class;
extern int* num_react_species;
extern int* num_products;
extern int* is_ec;
extern int* is_reverse;

extern int* HYP_RESTRICT reactant_1;
extern int* HYP_RESTRICT reactant_2;
extern int* HYP_RESTRICT reactant_3;
extern double* HYP_RESTRICT reactant_filter;

extern double* HYP_RESTRICT rate;
extern double* HYP_RESTRICT flux;
extern double* q_value;
extern double* p_0;
extern double* p_1;
extern double* p_2;
extern double* p_3;
extern double* p_4;
extern double* p_5;
extern double* p_6;

extern int** reactant_n;
extern int** reactant_z;
extern int** product_n;
extern int** product_z;
extern int** reactant_idx;
extern int** product_idx;

extern int* z;
extern int* n;

extern double* aa;
extern double* x;
extern double* y;
extern double* mass_excess;

extern double** partition_func;
extern double* partition_func_temp;

extern int f_plus_total;
extern int f_minus_total;

extern double* HYP_RESTRICT f_plus;
extern double* HYP_RESTRICT f_minus;
extern double* HYP_RESTRICT f_plus_factor;
extern double* HYP_RESTRICT f_minus_factor;
extern double* HYP_RESTRICT f_plus_sum;
extern double* HYP_RESTRICT f_minus_sum;
extern double* HYP_RESTRICT prefactor;

extern int* HYP_RESTRICT f_plus_max;
extern int* HYP_RESTRICT f_plus_min;
extern int* HYP_RESTRICT f_minus_max;
extern int* HYP_RESTRICT f_minus_min;
extern int* HYP_RESTRICT f_plus_isotope_cut;
extern int* HYP_RESTRICT f_minus_isotope_cut;
extern int* HYP_RESTRICT f_plus_num;
extern int* HYP_RESTRICT f_minus_num;
extern int* HYP_RESTRICT f_plus_isotope_idx;
extern int* HYP_RESTRICT f_minus_isotope_idx;
extern int* HYP_RESTRICT f_plus_map;
extern int* HYP_RESTRICT f_minus_map;
extern int** reaction_mask;

#ifdef __HYPERION_USE_SIMD
#include <immintrin.h>
#endif

#ifdef __HYPERION_USE_SIMD
extern __m512d _aa1;
extern __m512d _aa2;
#endif

#endif
