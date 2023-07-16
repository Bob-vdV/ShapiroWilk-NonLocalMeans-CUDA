add_executable(swilk_test
    swilktest.cpp
)
add_compile_options(-lm)

set_target_properties(swilk_test PROPERTIES
 CUDA_SEPARABLE_COMPILATION ON
 )

target_link_libraries(swilk_test 
    swilkLib
)