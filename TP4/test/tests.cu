#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <helper_cuda.h>



////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}



////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata, int blockSize)
{
    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // First, each thread loads data into shared memory

    temp[tid] = g_idata[tid];

    __syncthreads();

    // Find first power of 2 less then blockSize

    int m;
    for (m = 1; m < blockSize; m *= 2);
      m /= 2;

    if (tid + m < blockSize) {
        temp[tid] += temp[tid + m];
    }

    __syncthreads();

    // Next, we perform binary tree reduction

    for (int d = m / 2; d > 0; d /= 2) {
        __syncthreads();
        if (tid < d) {
            temp[tid] += temp[tid + d];
        }
    }

    // Finally, first thread puts result into global memory

    if (tid == 0) {
        g_odata[0] = temp[0];  
    }
}



int g_argc;
const char** g_argv;



//
// Unitary test to compare reductions
//

TEST(CudaTest, equality)
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *d_idata, *d_odata;

  // Initialise card

  findCudaDevice(g_argc, g_argv);

  // Start with only 1 thread block

  num_blocks   = 1;  
  num_threads  = 192;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  // Allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // Compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // Allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float)) );

  // Copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // Execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata, num_elements);
  getLastCudaError("reduction kernel execution failed");

  // Copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float),
                              cudaMemcpyDeviceToHost) );

  // Unitary test for reductions

  ASSERT_FLOAT_EQ(h_data[0], sum);

  // Check results

  printf("reduction error = %f\n",h_data[0]-sum);

  // Cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}



//
// Call of tests
//

int main(int argc, char **argv)
{
    g_argc = argc;
    g_argv = const_cast<const char**>(argv);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}