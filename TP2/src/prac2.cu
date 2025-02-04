////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

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
    ind += blockDim.x;      // shift pointer to next element
    // Version 2
    // ind += 1;

    y2   = rho*y1 + alpha*d_z[ind];
    // Version 1
    ind += blockDim.x;      // shift pointer to next element
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
        float z = curand_normal(&localState);  // Génère un z ~ N(0,1)
        sum += a * z * z + b * z + c;
    }

    x[idx] = sum / 200;
    d_states[idx] = localState;
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv)
{
  int     NPATH=9600000, h_N=100, SIZE=200, nblocks, nthreads, nsize;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2, h_result;
  float  *h_v, *h_x, *d_v, *d_z, *d_x;
  double  sum1, sum2;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, 2.0*h_N*NPATH/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);
  pathcalc<<<NPATH/128, 128>>>(d_z, d_v);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("pathcalc execution failed\n");
  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  // Calcul du volume de données transférées
  double data_read = 2.0 * h_N * NPATH * sizeof(float) / 1e9; // en Go
  double data_written = NPATH * sizeof(float) / 1e9; // en Go
  double total_data = data_read + data_written;

  // Calcul du taux de transfert effectif
  double execution_time = milli / 1000.0; // conversion en secondes
  double bandwidth = total_data / execution_time; // Go/s

  printf("Data read: %f GB\n", data_read);
  printf("Data written: %f GB\n", data_written);
  printf("Total data transferred: %f GB\n", total_data);
  printf("Effective memory bandwidth: %f GB/s\n", bandwidth);

  printf("\n========================================================= \n\n");

  // Beginning of computation of the average value of az² + bz + c

  // Set number of blocks, and threads per block

  nblocks  = 256;
  nthreads = 256;
  nsize    = nblocks*nthreads ;
  curandState *d_states;

  h_x = (float *)malloc(sizeof(float)*nsize);
  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float)*nsize));
  cudaMalloc(&d_states, nsize * sizeof(curandState));

  // Error gestion

  if (h_x == NULL)
  {
    printf("Erreur d'allocation mémoire\n");
    return -1;
  }

  // Execute the kernel

  generate_random<<<nblocks,nthreads>>>(d_states, d_x, time(NULL));

  // Copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  float temporary_sum = 0.0f;

  for (int n=0; n<nsize; n++) temporary_sum += h_x[n];

  h_result = temporary_sum/nsize;

  printf("Final result: %f\n", h_result);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  free(h_x);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );
  checkCudaErrors( cudaFree(d_x) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
