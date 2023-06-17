add_executable(swnlm_cuda
    swnlmcuda.cpp
)
add_compile_options(-lm -lstdc++fs)

target_link_libraries(swnlm_cuda
    ${OpenCV_LIBS}
    utilsLib
    swnlmCudaLib
    ${CUDA_LIBRARIES}
)

