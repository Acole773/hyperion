// IMPORTANT WARNING!!!
// There is no god here, nobody is watching over you...
// With every glance at this code, your mind loses its grasp on reality.
// Leave this place, and never come back.

#include "../core/store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <limits.h>

#define LABELSIZE 35

void rate_library_create(char* filename, int size) {

    reaction_label = malloc(size * sizeof(char*));

    rg_member_idx = malloc(size * sizeof(int));
    rg_class = malloc(size * sizeof(int));
    reaction_library_class = malloc(size * sizeof(int));
    num_react_species = malloc(size * sizeof(int));
    num_products = malloc(size * sizeof(int));
    is_ec = malloc(size * sizeof(int));
    is_reverse = malloc(size * sizeof(int));

    rate = malloc(size * sizeof(double));
    flux = malloc(size * sizeof(double));
    prefactor = malloc(size * sizeof(double));
    q_value = malloc(size * sizeof(double));
    p_0 = malloc(size * sizeof(double));
    p_1 = malloc(size * sizeof(double));
    p_2 = malloc(size * sizeof(double));
    p_3 = malloc(size * sizeof(double));
    p_4 = malloc(size * sizeof(double));
    p_5 = malloc(size * sizeof(double));
    p_6 = malloc(size * sizeof(double));

    reactant_n = malloc(size * sizeof(int*));
    reactant_z = malloc(size * sizeof(int*));
    product_n = malloc(size * sizeof(int*));
    product_z = malloc(size * sizeof(int*));
    reactant_1 = malloc(size * sizeof(int));
    reactant_2 = malloc(size * sizeof(int));
    reactant_3 = malloc(size * sizeof(int));
    reactant_idx = malloc(size * sizeof(int*));
    product_idx = malloc(size * sizeof(int*));

    char line[120];
    char reaction_token[LABELSIZE];
    double p0, p1, p2, p3, p4, p5, p6, q, sf;
    int i0, i1, i2, i3, i4, i5, i6;
    int ii[6];

    /*
    Read in the file line by line and parse into variables. The expected
    structure of each line is:
         double double double double double double double string
    ...each separated by a space, with no whitespace in the string.
    */

    int n = -1;
    int subindex = -1;

    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    fprintf(stderr, "CWD = %s\n", cwd);
    fprintf(stderr, "Trying to open file = %s\n", filename);
    fflush(stderr);

    FILE* file = fopen(filename, "r");
    if (!file) {
    	fprintf(stderr, "ERROR: failed to open %s\n", filename);
    	abort();
    }

    while (fgets(line, 120, file)) {
        subindex++;
        switch (subindex) {
        case 0:
            n++;
            sscanf(line, "%s %d %d %d %d %d %d %d %lf %lf", reaction_token, &i0,
                   &i1, &i2, &i3, &i4, &i5, &i6, &sf, &q);
            reaction_label[n] =
                malloc(sizeof(char) * (strlen(reaction_token) + 1));
            strcpy(reaction_label[n], reaction_token);
            rg_class[n] = i0;
            rg_member_idx[n] = i1;
            reaction_library_class[n] = i2;
            num_react_species[n] = i3;
            num_products[n] = i4;
            is_ec[n] = i5;
            is_reverse[n] = i6;
            prefactor[n] = sf;
            q_value[n] = q;
            break;
        case 1:
            sscanf(line, "%lf %lf %lf %lf %lf %lf %lf", &p0, &p1, &p2, &p3, &p4,
                   &p5, &p6);
            p_0[n] = p0;
            p_1[n] = p1;
            p_2[n] = p2;
            p_3[n] = p3;
            p_4[n] = p4;
            p_5[n] = p5;
            p_6[n] = p6;
            break;
        case 2:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            reactant_z[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_react_species[n]; mm++) {
                reactant_z[n][mm] = ii[mm];
            }
            break;
        case 3:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            reactant_n[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_react_species[n]; mm++) {
                reactant_n[n][mm] = ii[mm];
            }
            break;
        case 4:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            product_z[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_products[n]; mm++) {
                product_z[n][mm] = ii[mm];
            }
            break;
        case 5:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            product_n[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_products[n]; mm++) {
                product_n[n][mm] = ii[mm];
            }
            break;
        case 6:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            reactant_idx[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_react_species[n]; mm++) {
                reactant_idx[n][mm] = ii[mm];
            }
            break;
        case 7:
            sscanf(line, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
            product_idx[n] = malloc(sizeof(int) * 4);
            for (int mm = 0; mm < num_products[n]; mm++) {
                product_idx[n][mm] = ii[mm];
            }
            subindex = -1;
            break;
        }
    }
    num_reactions = n + 1;

    for (int i = 0; i < num_reactions; i++) {
        reactant_1[i] = reactant_idx[i][0];
        reactant_2[i] = reactant_idx[i][1];
        reactant_3[i] = reactant_idx[i][2];
    }

    fclose(file);

    return;
}

#define PF_ALLOC_CHUNK 24

void network_create(char* filename, int size) {

    z = malloc(size * sizeof(int));
    n = malloc(size * sizeof(int));

    aa = malloc(size * sizeof(double));
    x = malloc(size * sizeof(double));
    y = malloc(size * sizeof(double));
    mass_excess = malloc(size * sizeof(double));

    partition_func = calloc(size, sizeof(double*));
    iso_label = malloc(size * sizeof(char*));

    char line[60];
    char iso_symbol[5];
    int z_, n_, a_;
    double y_, mass_;
    double pf0, pf1, pf2, pf3, pf4, pf5, pf6, pf7;

    double temp[24] = {0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                       0.8f, 0.9f,  1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f,
                       4.0f, 4.5f,  5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    partition_func_temp = malloc(sizeof(double) * PF_ALLOC_CHUNK);
    for (int i = 0; i < 24; i++) {
        partition_func_temp[i] = temp[i];
    }

    int iso_idx = -1;
    int iso_subindex = 3;

    FILE* file = fopen(filename, "r");

    // Assume lines can contain up to 60 characters.
    while (fgets(line, 60, file)) {
        iso_subindex++;
        if (iso_subindex == 4) {
            iso_subindex = 0;
            iso_idx++;
            // Scan and parse a title line
            sscanf(line, "%s %d %d %d %lf %lf", iso_symbol, &a_, &z_, &n_, &y_,
                   &mass_);
            z[iso_idx] = z_;
            n[iso_idx] = n_;
            aa[iso_idx] = (double)a_;
            y[iso_idx] = y_;
            x[iso_idx] = aa[iso_idx] * y[iso_idx];
            mass_excess[iso_idx] = mass_;
            iso_label[iso_idx] =
                malloc(sizeof(char) * (strlen(iso_symbol) + 1));
            strcpy(iso_label[iso_idx], iso_symbol);
        } else {
            // Scan and parse a partitionition function line.
            sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf", &pf0, &pf1, &pf2,
                   &pf3, &pf4, &pf5, &pf6, &pf7);
            int tin = iso_subindex - 1;
            if (partition_func[iso_idx] == NULL) {
                partition_func[iso_idx] =
                    malloc(sizeof(double) * PF_ALLOC_CHUNK);
            }
            partition_func[iso_idx][8 * (tin)] = pf0;
            partition_func[iso_idx][8 * (tin) + 1] = pf1;
            partition_func[iso_idx][8 * (tin) + 2] = pf2;
            partition_func[iso_idx][8 * (tin) + 3] = pf3;
            partition_func[iso_idx][8 * (tin) + 4] = pf4;
            partition_func[iso_idx][8 * (tin) + 5] = pf5;
            partition_func[iso_idx][8 * (tin) + 6] = pf6;
            partition_func[iso_idx][8 * (tin) + 7] = pf7;
        }

        num_species = iso_idx + 1;
    }
    
    fclose(file);

    return;
}

int* temp_int1;
int* temp_int2;

void reaction_mask_create(void) {
    reaction_mask = malloc(sizeof(int*) * num_species);

    for (int i = 0; i < num_species; i++) {
        reaction_mask[i] = malloc(sizeof(int) * num_reactions);
    }

    int increment_plus = 0;
    int increment_minus = 0;

    for (int i = 0; i < num_species; i++) {
        int total = 0;
        int f_plus_num_ = 0;
        int f_minus_num_ = 0;

        // Loop over all possible reactions for this isotope, finding those that
        // change its population up (contributing to F+) or down (contributing
        // to F-).
        for (int j = 0; j < num_reactions; j++) {
            int l_total = 0;
            int r_total = 0;

            // Loop over reactants for this reaction
            for (int k = 0; k < num_react_species[j]; k++) {
                if (z[i] == reactant_z[j][k] && n[i] == reactant_n[j][k])
                    l_total++;
            }

            // Loop over products for this reaction
            for (int k = 0; k < num_products[j]; k++) {
                if (z[i] == product_z[j][k] && n[i] == product_n[j][k])
                    r_total++;
            }

            total = l_total - r_total;

            if (total > 0) { // Contributes to F- for this isotope
                f_minus_num_++;
                reaction_mask[i][j] = -total;
                temp_int2[increment_minus + f_minus_num_ - 1] = j;
            } else if (total < 0) { // Contributes to F+ for this isotope
                f_plus_num_++;
                reaction_mask[i][j] = -total;
                temp_int1[increment_plus + f_plus_num_ - 1] = j;
            } else { // Does not contribute to flux for this isotope
                reaction_mask[i][j] = 0;
            }
        }

        // Keep track of the total num of F+ and F- terms in the network for
        // all isotopes
        f_plus_total += f_plus_num_;
        f_minus_total += f_minus_num_;

        f_plus_num[i] = f_plus_num_;
        f_minus_num[i] = f_minus_num_;

        increment_plus += f_plus_num_;
        increment_minus += f_minus_num_;
    }

    return;
}

void data_init(void) {

    // Find for each isotope all reactions that change its population.  This
    // analysis of the network is required only once at the very beginning of
    // the calculation (provided that the network species and reactions remain
    // the same for the entire calculation). The work is done by the function
    // reaction_mask_create().

    // num of F+ and F- components for each isotope
    f_plus_num = malloc(sizeof(int) * num_species);
    f_minus_num = malloc(sizeof(int) * num_species);

    temp_int1 = calloc(num_species * num_reactions / 2, sizeof(int));
    temp_int2 = calloc(num_species * num_reactions / 2, sizeof(int));

    reaction_mask_create();

    // Create 1D arrays to hold non-zero F+ and F- for all reactions for all
    // isotopes, the arrays holding the species factors f_plus_factor
    // and f_minus_factor, and also arrays to hold their sums for each
    // isotope. Note that parseF() must be run first because it determines
    // f_plus_total and f_minus_total.

    f_plus = malloc(sizeof(double) * f_plus_total);
    f_minus = malloc(sizeof(double) * f_minus_total);
    f_plus_factor = malloc(sizeof(double) * f_plus_total);
    f_minus_factor = malloc(sizeof(double) * f_minus_total);
    f_plus_sum = malloc(sizeof(double) * num_species);
    f_minus_sum = malloc(sizeof(double) * num_species);

    // Arrays that hold the index of the boundary between different isotopes in
    // the f_plus and f_minus 1D arrays. Since f_plus_max and f_plus_min are
    // related, and likewise f_minus_max and f_minus_min are related, we will
    // only need to pass f_plus_max and f_minus_max to the kernel.

    f_plus_max = malloc(sizeof(int) * num_species);
    f_plus_min = malloc(sizeof(int) * num_species);
    f_minus_max = malloc(sizeof(int) * num_species);
    f_minus_min = malloc(sizeof(int) * num_species);

    // Create 1D arrays that will be used to map finite F+ and F- to the Flux
    // array.

    f_plus_isotope_cut = malloc(sizeof(int) * num_species);
    f_minus_isotope_cut = malloc(sizeof(int) * num_species);

    f_plus_isotope_idx = malloc(sizeof(int) * f_plus_total);
    f_minus_isotope_idx = malloc(sizeof(int) * f_minus_total);

    // Create 1D arrays that will hold the index of the isotope for the F+ or F-
    // term
    f_plus_map = malloc(sizeof(int) * f_plus_total);
    f_minus_map = malloc(sizeof(int) * f_minus_total);

    f_plus_isotope_cut[0] = f_plus_num[0];
    f_minus_isotope_cut[0] = f_minus_num[0];
    for (int i = 1; i < num_species; i++) {
        f_plus_isotope_cut[i] = f_plus_num[i] + f_plus_isotope_cut[i - 1];
        f_minus_isotope_cut[i] = f_minus_num[i] + f_minus_isotope_cut[i - 1];
    }

    int current_iso = 0;
    for (int i = 0; i < f_plus_total; i++) {
        f_plus_isotope_idx[i] = current_iso;
        if (i == (f_plus_isotope_cut[current_iso] - 1)) {
            current_iso++;
        }
    }

    current_iso = 0;
    for (int i = 0; i < f_minus_total; i++) {
        f_minus_isotope_idx[i] = current_iso;
        if (i == (f_minus_isotope_cut[current_iso] - 1))
            current_iso++;
    }

    for (int i = 0; i < f_plus_total; i++) {
        f_plus_map[i] = temp_int1[i];
    }

    for (int i = 0; i < f_minus_total; i++) {
        f_minus_map[i] = temp_int2[i];
    }

    // Populate the f_plus_min and f_plus_max arrays
    f_plus_min[0] = 0;
    f_plus_max[0] = f_plus_num[0] - 1;
    for (int i = 1; i < num_species; i++) {
        f_plus_min[i] = f_plus_max[i - 1] + 1;
        f_plus_max[i] = f_plus_min[i] + f_plus_num[i] - 1;
    }
    // Populate the f_minus_min and f_minus_max arrays
    f_minus_min[0] = 0;
    f_minus_max[0] = f_minus_num[0] - 1;
    for (int i = 1; i < num_species; i++) {
        f_minus_min[i] = f_minus_max[i - 1] + 1;
        f_minus_max[i] = f_minus_min[i] + f_minus_num[i] - 1;
    }

    // Populate the f_plus_factor and f_minus_factor arrays that
    // hold the factors counting the num of occurences of the species in the
    // reaction.  Note that this can only be done after parseF() has been run to
    // give reaction_mask[i][j].

    int temp_count_plus = 0;
    int temp_count_minus = 0;
    for (int i = 0; i < num_species; i++) {
        for (int j = 0; j < num_reactions; j++) {
            if (reaction_mask[i][j] > 0) {
                f_plus_factor[temp_count_plus] = (double)reaction_mask[i][j];
                temp_count_plus++;
            } else if (reaction_mask[i][j] < 0) {
                f_minus_factor[temp_count_minus] = -(double)reaction_mask[i][j];
                temp_count_minus++;
            }
        }
    }

    free(temp_int1);
    free(temp_int2);

    return;
}
