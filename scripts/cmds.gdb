# 1. Force the debugger to accept breakpoints on code not yet loaded (the GPU kernel)
set breakpoint pending on

# 2. Setup breakpoints for the host wrapper and GPU kernel
# Source [1]: The host entry point
break hyperion_burner_
break hyperion_burner_dev_kernel

# 3. Start the program
run

# --- Debugger stops at hyperion_burner_ (Host) ---
echo \n--- Reached Host Wrapper ---\n

# 4. Continue execution to reach the GPU kernel
continue

# --- Debugger stops at hyperion_burner_dev_kernel (Device) ---
echo \n--- Reached GPU Kernel ---\n

# 5. NOW you can inspect kernel-specific variables
print zone
print tid

# Inspect the suspected NaN source
print aa_zone
print xout_zone

# 6. Continue to the end
continue

quit


