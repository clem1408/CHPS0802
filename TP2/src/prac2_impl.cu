#include "prac2_impl.h"



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