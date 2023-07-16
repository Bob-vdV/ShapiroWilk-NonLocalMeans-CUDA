add_executable(cuTest
    cuTest.cu
)

set_target_properties(cuTest PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
 )

target_link_libraries(cuTest 
    cudaTest
    commonLib
)