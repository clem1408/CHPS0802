#include <gtest/gtest.h>
#include <helper_cuda.h>
#include "../src/prac2_impl.h"



__constant__ int   N, a = 1, b = 2, c = 3;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;



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
