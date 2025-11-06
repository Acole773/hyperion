#include "kill.h"
            
#include <stdlib.h>

void _killall_ptrs() {
    free(rg_member_idx);
    free(rg_class);
    free(reaction_library_class);
    free(num_react_species);
    free(num_products);
    free(is_ec);
    free(is_reverse);
    free(reactant_1);
    free(reactant_2);
    free(reactant_3);
    free(rate);
    free(flux);
    free(q_value);
    free(p_0);
    free(p_1);
    free(p_2);
    free(p_3);
    free(p_4);
    free(p_5);
    free(p_6);
    free(z);
    free(n);
    free(aa);
    free(x);
    free(y);
    free(mass_excess);
    free(partition_func_temp);
    free(f_plus);
    free(f_minus);
    free(f_plus_factor);
    free(f_minus_factor);
    free(f_plus_sum);
    free(f_minus_sum);
    free(prefactor);
    free(f_plus_max);
    free(f_plus_min);
    free(f_minus_max);
    free(f_minus_min);
    free(f_plus_isotope_cut);
    free(f_minus_isotope_cut);
    free(f_plus_num);
    free(f_minus_num);
    free(f_plus_isotope_idx);
    free(f_minus_isotope_idx);
    free(f_plus_map);
    free(f_minus_map);

    for (int i = 0; i < num_species; i++) {
        free(iso_label[i]);
        free(partition_func[i]);
        free(reaction_mask[i]);
    }
    for (int i = 0; i < num_reactions; i++) {
        free(reactant_idx[i]);
        free(reaction_label[i]);
        free(product_idx[i]);
        free(product_n[i]);
        free(product_z[i]);
        free(reactant_n[i]);
        free(reactant_z[i]);
    }
    free(reactant_n);
    free(reactant_z);
    free(product_n);
    free(product_z);
    free(iso_label);
    free(partition_func);
    free(reaction_mask);
    free(reactant_idx);
    free(product_idx);
    free(reaction_label);

    return;
}
