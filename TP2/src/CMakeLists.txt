# Add source files for separate executables
set(HOST_SOURCE prac2.cu)
set(DEVICE_SOURCE prac2_currand.cu)

# Add executables
add_executable(prac2 ${HOST_SOURCE})
add_executable(prac2_currand ${DEVICE_SOURCE})

target_link_libraries(prac2 PRIVATE ${CUDA_LIBRARIES} curand)
target_link_libraries(prac2_currand PRIVATE ${CUDA_LIBRARIES} curand)