////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////



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
// GPU routine: Reduction per block (partial sums stored in global memory)
////////////////////////////////////////////////////////////////////////

__global__ void reduction_per_block(float *g_odata, float *g_idata, int num_elements) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    temp[tid] = (idx < num_elements) ? g_idata[idx] : 0.0f;
    __syncthreads();

    // Binary tree reduction 
    for (int d = blockDim.x / 2; d > 0; d /= 2) {
        __syncthreads();
        if (tid < d) {
            temp[tid] += temp[tid + d];
        }
    }

    // Store the result of this block in global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = temp[0];
    }
}



////////////////////////////////////////////////////////////////////////
// GPU routine: Reduction using atomic addition
////////////////////////////////////////////////////////////////////////

__global__ void reduction_atomic(float *g_odata, float *g_idata, int num_elements)
{
  extern __shared__ float temp[];
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  temp[tid] = (idx < num_elements) ? g_idata[idx] : 0.0f;
  __syncthreads();

  // Binary tree reduction 
  for (int d = blockDim.x / 2; d > 0; d /= 2) {
      __syncthreads();
      if (tid < d) {
          temp[tid] += temp[tid + d];
      }
  }

  if (tid == 0) {
      atomicAdd(g_odata, temp[0]);
  }
}



////////////////////////////////////////////////////////////////////////
// GPU routine: Reduction using shuffle
////////////////////////////////////////////////////////////////////////

__global__ void reduction_shuffle(float *g_odata, float *g_idata, int num_elements)
{
    int tid = threadIdx.x; 
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load value in register

    float sum = (idx < num_elements) ? g_idata[idx] : 0.0f;

    // Reduction using shuffle instructions

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Use shared memory for the reduction

    __shared__ float partial_sums[32];  
    
    if (tid % warpSize == 0) {  
        partial_sums[tid / warpSize] = sum;
    }
    
    __syncthreads();

    // Final reduction

    if (tid < warpSize) {
        sum = (tid < blockDim.x / warpSize) ? partial_sums[tid] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (tid == 0) {
            g_odata[blockIdx.x] = sum;  
        }
    }
}



////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks = 4;  
  int num_threads = 256;
  int num_elements = num_blocks * num_threads;
  int mem_size = sizeof(float) * num_elements;

  float *h_data, *d_idata, *d_odata, *d_result;

  // Allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
  
  for(int i = 0; i < num_elements; i++) {
      h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));
  }

  // Compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // Allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void**)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void**)&d_odata, sizeof(float) * num_blocks));
  checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(float)));
  
  // Copy host memory to device input array
  
  checkCudaErrors(cudaMemset(d_result, 0, sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Reduction per block

  cudaEventRecord(start);
  reduction_per_block<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(d_odata, d_idata, num_elements);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Reduction per block execution time (ms): %f\n", milli);

  // Copy partial results and compute final sum on CPU

  float *h_odata = (float*) malloc(sizeof(float) * num_blocks);
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));
  float final_sum = 0.0f;
  for (int i = 0; i < num_blocks; i++) {
      final_sum += h_odata[i];
  }

  // Check results

  printf("Reduction per block error = %f\n", final_sum - sum);

  // Atomic reduction

  cudaEventRecord(start);
  reduction_atomic<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(d_result, d_idata, num_elements);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Atomic reduction execution time (ms): %f\n", milli);

  // Check results

  float h_result;
  checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Atomic reduction error = %f\n", h_result - sum);

  // Shuffle reduction

  cudaEventRecord(start);
  reduction_shuffle<<<num_blocks, num_threads>>>(d_odata, d_idata, num_elements);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("Shuffle reduction execution time (ms): %f\n", milli);

  // Copy partial results and compute final sum on CPU

  h_odata = (float*) malloc(sizeof(float) * num_blocks);
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost));
  
  final_sum = 0.0f;
  for (int i = 0; i < num_blocks; i++) {
      final_sum += h_odata[i];
  }

  // Check results

  printf("Reduction (shuffle) per block error = %f\n", final_sum - sum);

  // Cleanup memory

  free(h_data);
  free(h_odata);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));
  checkCudaErrors(cudaFree(d_result));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
