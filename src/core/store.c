#include "store.h"

int num_reactions;
int num_species;

char** reaction_label;
char** iso_label;

// TODO: restrict properly
int* restrict rg_member_idx;
int* rg_class;
int* reaction_library_class;
int* num_react_species;
int* num_products;
int* is_ec;
int* is_reverse;

int* restrict reactant_1;
int* restrict reactant_2;
int* restrict reactant_3;

double* restrict rate;
double* restrict flux;
double* q_value;
double* p_0;
double* p_1;
double* p_2;
double* p_3;
double* p_4;
double* p_5;
double* p_6;

int** reactant_n;
int** reactant_z;
int** product_n;
int** product_z;
int** reactant_idx;
int** product_idx;

int* z;
int* n;

double* aa;
double* x;
double* y;
double* mass_excess;

double** partition_func;
double* partition_func_temp;

int f_plus_total;
int f_minus_total;

double* restrict f_plus;
double* restrict f_minus;
double* restrict f_plus_factor;
double* restrict f_minus_factor;
double* restrict f_plus_sum;
double* restrict f_minus_sum;
double* restrict prefactor;

int* restrict f_plus_max;
int* restrict f_plus_min;
int* restrict f_minus_max;
int* restrict f_minus_min;
int* restrict f_plus_isotope_cut;
int* restrict f_minus_isotope_cut;
int* restrict f_plus_num;
int* restrict f_minus_num;
int* restrict f_plus_isotope_idx;
int* restrict f_minus_isotope_idx;
int* restrict f_plus_map;
int* restrict f_minus_map;

int** reaction_mask;

// TODO: For GPU (this is unacceptably terrible for a number of reasons: mainly
// that it cannot do more than 1 at a time, but we'll fix that later, I just
// want it to fucking work right now so that I can do other stuff)
void** args;
