#include "store.h"
#include "restrict.h"

int num_reactions;
int num_species;

char** reaction_label;
char** iso_label;

int* HYP_RESTRICT rg_member_idx;
int* rg_class;
int* reaction_library_class;
int* num_react_species;
int* num_products;
int* is_ec;
int* is_reverse;

int* HYP_RESTRICT reactant_1;
int* HYP_RESTRICT reactant_2;
int* HYP_RESTRICT reactant_3;

double* HYP_RESTRICT reactant_filter;

double* HYP_RESTRICT rate;
double* HYP_RESTRICT flux;
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

double* HYP_RESTRICT f_plus;
double* HYP_RESTRICT f_minus;
double* HYP_RESTRICT f_plus_factor;
double* HYP_RESTRICT f_minus_factor;
double* HYP_RESTRICT f_plus_sum;
double* HYP_RESTRICT f_minus_sum;
double* HYP_RESTRICT prefactor;

int* HYP_RESTRICT f_plus_max;
int* HYP_RESTRICT f_plus_min;
int* HYP_RESTRICT f_minus_max;
int* HYP_RESTRICT f_minus_min;
int* HYP_RESTRICT f_plus_isotope_cut;
int* HYP_RESTRICT f_minus_isotope_cut;
int* HYP_RESTRICT f_plus_num;
int* HYP_RESTRICT f_minus_num;
int* HYP_RESTRICT f_plus_isotope_idx;
int* HYP_RESTRICT f_minus_isotope_idx;
int* HYP_RESTRICT f_plus_map;
int* HYP_RESTRICT f_minus_map;

int** reaction_mask;


