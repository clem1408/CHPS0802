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
