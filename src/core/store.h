#ifndef __STORE_H
#define __STORE_H

extern int num_reactions;
extern int num_species;

extern char** reaction_label;
extern char** iso_label;

// TODO: restrict properly
extern int* restrict rg_member_idx;
extern int* rg_class;
extern int* reaction_library_class;
extern int* num_react_species;
extern int* num_products;
extern int* is_ec;
extern int* is_reverse;

extern int* restrict reactant_1;
extern int* restrict reactant_2;
extern int* restrict reactant_3;

extern double* restrict rate;
extern double* restrict flux;
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

extern double* restrict f_plus;
extern double* restrict f_minus;
extern double* restrict f_plus_factor;
extern double* restrict f_minus_factor;
extern double* restrict f_plus_sum;
extern double* restrict f_minus_sum;
extern double* restrict prefactor;

extern int* restrict f_plus_max;
extern int* restrict f_plus_min;
extern int* restrict f_minus_max;
extern int* restrict f_minus_min;
extern int* restrict f_plus_isotope_cut;
extern int* restrict f_minus_isotope_cut;
extern int* restrict f_plus_num;
extern int* restrict f_minus_num;
extern int* restrict f_plus_isotope_idx;
extern int* restrict f_minus_isotope_idx;
extern int* restrict f_plus_map;
extern int* restrict f_minus_map;
extern int** reaction_mask;

// For GPU
extern void** args;

#ifdef __HYPERION_USE_SIMD
#include <immintrin.h>
#endif

#ifdef __HYPERION_USE_SIMD
extern __m512d _aa1;
extern __m512d _aa2;
#endif

#endif
