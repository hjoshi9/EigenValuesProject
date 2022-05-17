# EigenValuesProject
The files given here contain C code for implementation of Power iteration method and Jacobi diagonalization method. 
Each file conists of serial implementation as well a parallel implementation using OPENMP directives reflected on the title of the file (i.e. main_gpu = GPU implementation using OPENMP).

Usage:
After complitaion, invoke the created applications with as:

main_omp : ./<omp_app> <size of matrix> <mode> <number of threads>

main_gpu : ./<gpu_app> <mode> <size of matrix> <mode>

mian_gpu_omp : ./<omp+gpu_app> <mode> <size of matrix>

where mode = 1 for Power iteration method, 
             2 for Jacobi method