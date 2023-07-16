add_compile_options(-lm -lstdc++fs)

add_executable(eval
    eval.cpp
)

set_target_properties(eval PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories( eval PRIVATE ${ROOT_SOURCE_DIR}/images)

target_link_libraries(eval 
    ${OpenCV_LIBS}
    cnlmLib
    cnlmCudaLib
    swnlmLib
    swnlmCudaLib
    utilsLib
)
