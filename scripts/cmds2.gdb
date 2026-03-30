set breakpoint pending on
set pagination off
set print pretty on
set print elements 8
set print address off

# --------------------------------------------------
# Break in host wrapper BEFORE kernel launch
# --------------------------------------------------
break hyperion_burner_kernel:129

run

echo \n--- Reached Host Wrapper ---\n

# Inspect launch parameters (optional but useful)
print zones

# Step to kernel launch line
next

echo \n--- About to launch kernel ---\n

# Optional: catch stub (host-side launch wrapper)
break __device_stub__hyperion_burner_dev_kernel

continue

echo \n--- Kernel launch attempted ---\n

# Step over kernel launch (this is where crash occurs)
next

echo \n--- Returned from kernel (if no crash) ---\n

continue
quit
