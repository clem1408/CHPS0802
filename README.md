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