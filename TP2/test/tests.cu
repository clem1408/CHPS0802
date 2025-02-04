#include <gtest/gtest.h>
#include <helper_cuda.h>
#include "../src/prac2_impl.h"

//
// Unitary test for a*z + b*z + c ~= a + c
//
TEST(CudaTest, ValueCloseness)
{
    int nblocks = 256, nthreads = 256;
    int nsize = nblocks * nthreads;
    
    int a = 1, b = 2, c = 3;  // Définition locale des constantes

    float *h_x = (float *)malloc(sizeof(float) * nsize);
    float *d_x;
    curandState *d_states;

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float) * nsize));
    checkCudaErrors(cudaMalloc((void **)&d_states, nsize * sizeof(curandState)));

    // Lancer le kernel en passant `a`, `b`, `c` en arguments
    generate_random<<<nblocks, nthreads>>>(d_states, d_x, time(NULL), a, b, c);
    checkCudaErrors(cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost));

    // Calcul de la moyenne de a*z² + b*z + c
    float sum = 0.0f;
    for (int i = 0; i < nsize; i++)
        sum += h_x[i];

    float mean_value = sum / nsize;
    float expected_value = a + c;

    // Vérification que la valeur moyenne est proche de a + c
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
