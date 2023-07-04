add_executable(cnlm_cuda
    cnlmcuda.cpp
)

target_link_libraries(cnlm_cuda
    utilsLib
    cnlmCudaLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

