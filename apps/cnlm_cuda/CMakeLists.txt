add_executable(cnlm_cuda
    cnlmcuda.cpp
)

set_target_properties(cnlm_cuda PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cnlm_cuda
    utilsLib
    cnlmCudaLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

