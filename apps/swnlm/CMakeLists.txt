add_executable(swnlm
    swnlm.cpp
    swilk.hpp
    swilk.cpp 
)
add_compile_options(-lm)

target_link_libraries(swnlm ${OpenCV_LIBS} )