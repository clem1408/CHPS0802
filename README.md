# CHPS0802

This repository contains all the TPs of **CHPS0802**.

<hr></hr>

## How to compile and run the codes

To run these programs, if you make changes to the `CMakeLists.txt` file, you first need to execute these commands while being at the root directory of the TP you want to run:

```
cd build/
cmake ..
```

After this, while still in the `/build/` directory, if you want to run the code, you need to execute the following commands (we'll assume we are working on TP1 for this example):

```
make
```

You will see (in the case of TP1) that two executables are created: one for the unit tests and one for the main TP source file. You then simply need to run the one you want:

```
./runUnitTests
./tp1
```

To clean up the executables, you can run `make clean`.

**Note: Please note that I have added a bash script to run these programs more easily. Simply execute the following commands (from the root of the TP) depending on whether you want to run unitary tests or regular TP files:**

```
cd bash/
sh src.sh
sh unitaryTests.sh
```

<hr>

## TP1

This first TP is about adding two vectors on the GPU. We first add them on the CPU to generate "control" values, and then we add them on the GPU. I have included a unit test in the `/test/` directory that verifies the correctness of the vector addition performed on the GPU.

Here's a screenshot of the execution:

![image](https://github.com/user-attachments/assets/6c9beb54-f012-4a5c-ab1e-0056589c51e6)

<hr>

## TP2

In this second TP, we noticed that memory management is very important. Indeed, during the tests where we had to comment/uncomment certain parts of the code, we were able to test two versions of cache access by the threads.

In the first version, the first four threads retrieve the first cache line (32 bytes), and each one uses 25% of this line (as they work on a contiguous space), meaning they fully utilize the entire cache line.

However, in the second version, they only take 25% of each cache line! As a result, only 25% of the cache lines are effectively used, leading to a significantly higher number of cache access operations.

This demonstrates that managing cache access is crucial for performance optimization when programming in CUDA.

In question 5, it is requested to calculate the memory bandwidth of our GPU, which I have implemented but is not functioning correctly. As shown in the screenshot below, my GPU appears to be more than 38 times more powerful than it should be. The technical specifications of my GPU (4050 portable edition) should be 192 GB/s.

![image](https://github.com/user-attachments/assets/3f6d7f2d-da11-4864-8ce4-0b5360d07463)

For question 6, I have implemented the generation of random numbers on the GPU to calculate `az² + bz + c`. Each thread generates 200 random numbers and writes the average of these numbers into an array. Then, I take this array and calculate its average, and I find that the value is close to `a + c`, as specified in the problem statement.

A unit test has been set up to verify that this value is close to `a + c` (`a = 1` and `c = 3`):

![image](https://github.com/user-attachments/assets/ac66d33f-8571-4381-8612-2c7d969139a5)
![image](https://github.com/user-attachments/assets/0aad7a49-1857-4287-bb2d-0ac3272a1eb8)

<hr>

## TP3

In this third TP, we learn more about block size and how it can help us optimize code.

I have created a unit test to check that the GPU code and the Gold code produce the same output.

Grid size (`512 * 512 * 512`):
<br>
![image](https://github.com/user-attachments/assets/c55d1a49-f7d6-403d-95cf-2c96bae99419)

Grid size (`1024 * 1024 * 1024`):
<br>
![image](https://github.com/user-attachments/assets/06d5691e-361a-4acf-b49b-724dea4d7e1b)

After testing multiple block dimensions for `laplace3d.cu`, I found that the optimal values for both x and y dimensions are 32, as shown in the table below.

| x  | y  | Result (ms) |
|----|----|------------|
| 1  | 1  | 420.9      |
| 1  | 2  | 258.8      |
| 1  | 4  | 183.8      |
| 1  | 8  | 181.6      |
| 1  | 16 | 284.9      |
| 1  | 32 | 404.0      |
| 2  | 1  | 124.7      |
| 4  | 1  | 124.4      |
| 8  | 1  | 72.2       |
| 16 | 1  | 42.5       |
| 32 | 1  | 50.7       |
| 2  | 2  | 154.2      |
| 4  | 4  | 75.0       |
| 8  | 8  | 60.3       |
| 16 | 16 | 24.1       |
| **32** | **32** | **23.8**       |

After testing multiple block dimensions for `laplace3d_new.cu`, I found that the optimal values for x, y, and z dimensions are 16, 4, and 4, respectively, as shown in the table below.

Note: These values were provided by the teacher.

| x  | y  | z  | Result (ms) |
|----|----|----|------------|
| 1  | 1  | 1  | 2034.5     |
| 2  | 1  | 1  | 1034.7     |
| 1  | 2  | 1  | 1032.2     |
| 1  | 1  | 2  | 1043.1     |
| 2  | 2  | 2  | 305.1      |
| 4  | 1  | 1  | 551.5      |
| 1  | 4  | 1  | 566.0      |
| 1  | 1  | 4  | 533.8      |
| 4  | 4  | 4  | 58.7       |
| 8  | 1  | 1  | 312.7      |
| 1  | 8  | 1  | 293.0      |
| 1  | 1  | 8  | 299.8      |
| 8  | 8  | 8  | 40.6       |
| 16 | 1  | 1  | 168.3      |
| 1  | 16 | 1  | 172.9      |
| 1  | 1  | 16 | 228.3      |
| 16 | 2  | 2  | 53.4       |
| 2  | 16 | 2  | 101.0      |
| 2  | 2  | 16 | 101.6      |
| **16** | **4**  | **4**  | **21.7**      |
| 4  | 16 | 4  | 58.6       |
| 4  | 4  | 16 | 60.9       |
| 16 | 8  | 8  | 38.8       |
| 8  | 16 | 8  | 44.9       |
| 8  | 8  | 16 | 45.1       |
| 32 | 1  | 1  | 91.7       |
| 1  | 32 | 1  | 193.8      |
| 1  | 1  | 32 | 603.8      |

<hr>

## TP4

In this fourth TP, we learn more about reductions and how to implement them in several ways.

I have created a unit test to check that the GPU reduction is correctly performed.

![image](https://github.com/user-attachments/assets/7cd05b56-2dae-4998-abd8-9bc83bcd13f0)

I added the small piece of code provided in the assignment to make the reduction work for a number of threads that is not a power of 2:

```cpp
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
```

Finally, I have implemented the two reduction methods mentioned in question 4, as well as the shuffle-based reduction, and measured their execution times.

![image](https://github.com/user-attachments/assets/833c7603-8a42-427f-8e31-0e9d527f603b)

