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
__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float) threadIdx.x;

  //printf("Current tid : %d\n", tid);
}

// Kernel that add values of two vectors
__global__ void add_vectors_kernel(float *x, float *y)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = x[tid] + y[tid];
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
void add_floats_vectors(float *vector, float *vector2, int size)
{
    for (int i = 0; i < size; i++)
  	{
      printf("[HOST] Résultat : %f\n", vector[i] + vector2[i]);
    }
}



//
// Main code
//

int main(int argc, const char **argv)
{
  float *h_x, *h_v, *h_v2, *d_x, *d_v, *d_v2;
  int   nblocks, nthreads, nsize, n; 

  // Initialise card

  findCudaDevice(argc, argv);

  // Set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // Allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));

  // Allocation des deux vecteurs

  h_v = (float *)malloc(nsize*sizeof(float));
  h_v2 = (float *)malloc(nsize*sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_v, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v2, nsize*sizeof(float)));

  // Gestion des mallocs

  if (h_x == NULL || h_v == NULL || h_v2 == NULL) {
    printf("Erreur d'allocation mémoire\n");
    return -1;
  }

  // Init des deux vecteurs

  fill_vector_randomly(h_v, nsize);
  fill_vector_randomly(h_v2, nsize);

  // Addition des deux vecteurs sur le CPU

  printf("============================\n");

  add_floats_vectors(h_v, h_v2, nsize);

  printf("============================\n");

  // Meccopy de mes deux vecteurs

  checkCudaErrors( cudaMemcpy(d_v,h_v,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_v2,h_v2,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );

  // Execute kernel

  add_vectors_kernel<<<nblocks,nthreads>>>(d_v, d_v2);
  getLastCudaError("add_vectors_kernel execution failed\n");

  // Copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_v,d_v,nsize*sizeof(float),
               cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf("[DEVICE] Résultat : %f\n",h_v[n]);

  printf("============================\n");

  // Execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  getLastCudaError("my_first_kernel execution failed\n");

  // Copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // Free memory

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_v2));
  free(h_x);
  free(h_v);
  free(h_v2);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
