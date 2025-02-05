# CMake minimal requiered version
cmake_minimum_required(VERSION 3.18)

# TP2
project(TP2 LANGUAGES CXX CUDA)

# Turn on warnings for the compilator
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Set cuda architectures for my 3070 ti and my 4050
set(CMAKE_CUDA_ARCHITECTURES 80)

# Include headers
include_directories(include)

# Add source files for separate executables
set(HOST_SOURCE src/prac2.cu src/prac2_impl.cu)
set(DEVICE_SOURCE src/prac2_currand.cu)

# Add executables
add_executable(prac2 ${HOST_SOURCE})
add_executable(prac2_currand ${DEVICE_SOURCE})

target_link_libraries(prac2 PRIVATE ${CUDA_LIBRARIES} curand)
target_link_libraries(prac2_currand PRIVATE ${CUDA_LIBRARIES} curand)

################
#Unitary tests
###############
enable_testing()

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        DOWNLOAD_EXTRACT_TIMESTAMP true
)

# Un-comment if issues with GoogleTest
#FetchContent_MakeAvailable(googletest)

# Creation of test executable
add_executable(runUnitTests test/tests.cu src/prac2_impl.cu)

# Link with GoogleTest
target_link_libraries(runUnitTests PRIVATE gtest gtest_main)

# Adding the test(s) in CMake
add_test(NAME value_closeness COMMAND runUnitTests)
