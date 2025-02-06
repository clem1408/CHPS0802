# Add source files for separate executables
set(FIRST laplace3d.cu laplace3d_gold.cpp)
set(SECOND laplace3d_new.cu laplace3d_gold.cpp)

# Add executables
add_executable(laplace3d ${FIRST})
add_executable(laplace3d_new ${SECOND})