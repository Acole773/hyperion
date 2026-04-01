#include "paths.h"
#include <stddef.h>   // for NULL

#ifndef HYPERION_DATA_DIR
#define HYPERION_DATA_DIR NULL
#endif

const char* hyperion_data_dir = HYPERION_DATA_DIR;
