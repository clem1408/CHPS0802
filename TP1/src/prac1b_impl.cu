//
// Include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>



//
// Kernel routine
// 

// Basic kernel
inline __global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float) threadIdx.x;

  //printf("Current tid : %d\n", tid);
}

// Kernel that add values of two vectors
__global__ void add_vectors_kernel(float *x, float *y, float *z)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  z[tid] = x[tid] + y[tid];
}



//
// Host functions
//

// Function to fill a vector with randoms floats
void fill_vector_randomly(float *vector, int size)
{
    // Set the seed
    srand((unsigned int)time(NULL));

    for (int i = 0; i < size; i++)
    {
        vector[i] = (float)rand() / RAND_MAX; // Generate a float in [0,1]
    }
}

// Function to add to vector on host for control
void add_floats_vectors(float *vector, float *vector2, float *vector3, int size)
{
    for (int i = 0; i < size; i++)
  	{
      vector3[i] = vector[i] + vector2[i];
      printf("[HOST] Résultat : %f\n", vector3[i]);
    }
}