#include "paths.h"
#include <limits.h>
#include <stdio.h>
#include "init.h"
#include "store.h"
#include "../parse-data/parser.h"


/* stringify helper */
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#if SIZE == 16
#define NETWORK_SIZE 16
#define NUM_REACTIONS 48
#define NETWORK_FILE "data/network-alpha.txt"
#define RATE_FILE "data/ratelibrary-alpha.txt"
#endif // __SIZE_16

#if SIZE == 150
#define NETWORK_SIZE 150
#define NUM_REACTIONS 1604
#define NETWORK_FILE "data/network-150.txt"
#define RATE_FILE "data/ratelibrary-150.txt"
#endif // __SIZE_150

#if SIZE == 365
#define NETWORK_SIZE 365
#define NUM_REACTIONS 4395
#define NETWORK_FILE "data/network-365.txt"
#define RATE_FILE "data/ratelibrary-365.txt"
#endif // __SIZE_365

void hyperion_init_() {
    char rate_path[PATH_MAX];
    char network_path[PATH_MAX];

    snprintf(rate_path, sizeof(rate_path),
             "%s/%s", hyperion_data_dir, RATE_FILE);

    snprintf(network_path, sizeof(network_path),
             "%s/%s", hyperion_data_dir, NETWORK_FILE);

    fprintf(stderr, "INIT: rate_library_create(%s)\n", rate_path);
    fprintf(stderr, "INIT: network_create(%s)\n", network_path);
    fflush(stderr);

    rate_library_create(rate_path, NUM_REACTIONS);
    network_create(network_path, NETWORK_SIZE);

    rate_library_fixup_unused_reactants();

    data_init();

    return;
}
