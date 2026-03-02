set breakpoint pending on
set pagination off
set print pretty on
set print elements 8
set print address off

break hyperion_burner_
break hyperion_burner_dev_kernel

run

echo \n--- Reached Host Wrapper ---\n
continue

echo \n--- Reached GPU Kernel ---\n

# Lock to one lane
lane 0

# Print execution identity
print blockIdx.x
print threadIdx.x

# Print zone
print zone

# Inspect data
print aa[zone]
print xout[zone][0]
print xout[zone][1]

# NaN checks
print isnan(xout[zone][0])
print isnan(xout[zone][1])

continue
quit
