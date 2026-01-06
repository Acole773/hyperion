#include "bn_burner_gpu.h"

int hip_backend_finalize(void)
{
    hip_killall_device_ptrs();
    return 0;
}

