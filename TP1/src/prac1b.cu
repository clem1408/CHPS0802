//
// Include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>
#include "../src/prac1b_impl.cu"



//
// Main code
//

int main(int argc, const char **argv)
{
  float *h_x, *h_v, *h_v2, *h_v3, *d_x, *d_v, *d_v2, *d_v3;
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

  // Allocate the three vectors

  h_v = (float *)malloc(nsize*sizeof(float));
  h_v2 = (float *)malloc(nsize*sizeof(float));
  h_v3 = (float *)malloc(nsize*sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_v, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v2, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v3, nsize*sizeof(float)));

  // Mallocs gestion

  if (h_x == NULL || h_v == NULL || h_v2 == NULL) 
  {
    printf("Erreur d'allocation mémoire\n");
    return -1;
  }

  // Init the two vectors

  fill_vector_randomly(h_v, nsize);
  fill_vector_randomly(h_v2, nsize);

  // Adding the two vectors on the cpu

  printf("============================\n");

  add_floats_vectors(h_v, h_v2, h_v3, nsize);

  printf("============================\n");

  // Meccopy of the two vectors

  checkCudaErrors( cudaMemcpy(d_v,h_v,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_v2,h_v2,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );

  // Execute kernel

  add_vectors_kernel<<<nblocks,nthreads>>>(d_v, d_v2, d_v3);
  getLastCudaError("add_vectors_kernel execution failed\n");

  // Copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_v3,d_v3,nsize*sizeof(float),
               cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf("[DEVICE] Résultat : %f\n",h_v3[n]);

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
  checkCudaErrors(cudaFree(d_v3));
  free(h_x);
  free(h_v);
  free(h_v2);
  free(h_v3);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
