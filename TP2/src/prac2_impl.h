#ifndef PRAC2_IMPL_H
#define PRAC2_IMPL_H



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>
#include <curand_kernel.h>



__global__ void pathcalc(float *d_z, float *d_v);
__global__ void generate_random(curandState *d_states, float *x, unsigned long seed);



#endif // PRAC2_IMPL_H