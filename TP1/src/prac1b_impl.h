#ifndef PRAC1B_IMPL_H
#define PRAC1B_IMPL_H

void fill_vector_randomly(float *vector, int size);
void add_floats_vectors(float *vector, float *vector2, float *vector3, int size);
__global__ void add_vectors_kernel(float *x, float *y, float *z);

#endif // PRAC1B_IMPL_H