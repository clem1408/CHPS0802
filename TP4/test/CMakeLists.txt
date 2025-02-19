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

set(ALL tests.cu)

# Creation of test executable
add_executable(tests ${ALL})

# Link with GoogleTest
target_link_libraries(tests PRIVATE gtest gtest_main)

# Adding the test(s) in CMake
add_test(NAME equality COMMAND tests)