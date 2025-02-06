#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N, a = 1, b = 2, c = 3;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;



////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;
  int   ind;

  // Move array pointers to correct position

  // Version 1
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  // Version 2
  //ind = 2*N*threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // Path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++)
  {
    y1   = d_z[ind];

    // Version 1
    ind += blockDim.x;      // Shift pointer to next element
    // Version 2
    // ind += 1;

    y2   = rho*y1 + alpha*d_z[ind];
    // Version 1
    ind += blockDim.x;      // Shift pointer to next element
    // Version 2
    // ind += 1;

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // Put payoff value into device array

  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f )
    payoff = exp(-r*T);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
}

__global__ void generate_random(curandState *d_states, float *x, unsigned long seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

    curand_init(seed, idx, 0, &d_states[idx]);

	curandState localState = d_states[idx];

    float sum = 0.0f;

    for (int i = 0; i < 200; i++) {
        float z = curand_normal(&localState);  // Generate a z in [0,1]
        sum += a * z * z + b * z + c;
    }

    x[idx] = sum / 200;
    d_states[idx] = localState;
}



//
// Unitary test for a*z + b*z + c ~= a + c
//

TEST(CudaTest, ValueCloseness)
{
    int nblocks = 256, nthreads = 256;
    int nsize = nblocks * nthreads;
    
    float *h_x = (float *)malloc(sizeof(float) * nsize);
    float *d_x;
    curandState *d_states;

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float) * nsize));
    checkCudaErrors(cudaMalloc((void **)&d_states, nsize * sizeof(curandState)));

    // Give the constant by the way of cudaMemcpyToSymbol
    int h_a = 1, h_b = 2, h_c = 3;
    cudaMemcpyToSymbol(a, &h_a, sizeof(int));
    cudaMemcpyToSymbol(b, &h_b, sizeof(int));
    cudaMemcpyToSymbol(c, &h_c, sizeof(int));

    // Start the kernel
    generate_random<<<nblocks, nthreads>>>(d_states, d_x, time(NULL));
    checkCudaErrors(cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost));

    // Computing the average of a*z² + b*z + c
    float sum = 0.0f;
    for (int i = 0; i < nsize; i++)
        sum += h_x[i];

    float mean_value = sum / nsize;
    float expected_value = h_a + h_c;

    // Testing if value is close to a + c
    EXPECT_NEAR(mean_value, expected_value, 0.1) << "La moyenne de a*z² + b*z + c est incorrecte !";

    free(h_x);
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_states));
}



//
// Call of tests
//

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
