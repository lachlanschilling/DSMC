# DSMC
Implementation of a primitive collision routine in MATLAB and CUDA C. Documentation of the MATLAB version is available.

mainDSMC_debug.m is a MATLAB implementation with debugging functionality. It computes simulations sequentially.

mainDSMC_parallel_v2.m is a MATLAB implementation that computes simulations in parallel on the CPU.

main_dsmc_singlesim.cu is a CUDA C implementation that computes one simulation in parallel on the GPU.

For more information about the collision routine, visit: http://www.algarcia.org/Pubs/DSMC97.pdf
