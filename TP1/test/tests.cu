#include <gtest/gtest.h>
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



//
// Unitary test for the addidion of 2 vectors
//

TEST(CudaTest, VectorAddition) 
{
  float *h_v, *h_v2, *h_v3, *d_v, *d_v2, *d_v3;
  int   nblocks, nthreads, nsize, n;

  // Set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // Allocations of the three vectors

  h_v = (float *)malloc(nsize*sizeof(float));
  h_v2 = (float *)malloc(nsize*sizeof(float));
  h_v3 = (float *)malloc(nsize*sizeof(float));

  checkCudaErrors(cudaMalloc((void **)&d_v, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v2, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v3, nsize*sizeof(float)));

  // Mallocs gestion

  if (h_v == NULL || h_v2 == NULL || h_v3 == NULL) 
  {
    printf("Erreur d'allocation mémoire\n");
  }

  // Init of the two vectors

  fill_vector_randomly(h_v, nsize);
  fill_vector_randomly(h_v2, nsize);

  // Addition of the two vectors on the cpu

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

  for (n=0; n<nsize; n++) EXPECT_EQ(h_v3[n], h_v[n] + h_v2[n]);

  printf("============================\n");

  // Free memory

  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_v2));
  checkCudaErrors(cudaFree(d_v3));
  free(h_v);
  free(h_v2);
  free(h_v3);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}



//
// Call of tests
//

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}