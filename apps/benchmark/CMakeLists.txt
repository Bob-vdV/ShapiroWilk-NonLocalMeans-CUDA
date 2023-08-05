add_compile_options(-lm -lstdc++fs)

add_executable(benchmark
    benchmark.cpp
)

set_target_properties(benchmark PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories( benchmark PRIVATE ${ROOT_SOURCE_DIR}/images)

target_link_libraries(benchmark 
    ${OpenCV_LIBS}
    cnlmLib
    cnlmCudaLib
    swnlmLib
    swnlmCudaLib
    utilsLib
)
