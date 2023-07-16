add_executable(swnlm
    swnlm.cpp
)
add_compile_options(-lm)

set_target_properties(swnlm PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
 )

target_link_libraries(swnlm 
    ${OpenCV_LIBS}
    swnlmLib
    utilsLib
)