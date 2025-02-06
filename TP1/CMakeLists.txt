# CMake minimal requiered version
cmake_minimum_required(VERSION 3.18)

# TP1
project(TP1 LANGUAGES CXX CUDA)

# Turn on warnings for the compilator
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Set cuda architectures for my 3070 ti and my 4050
#set(CMAKE_CUDA_ARCHITECTURES 80)

# Include headers
include_directories(include)

add_subdirectory(test/)
add_subdirectory(src/)