add_executable(swnlm
    swnlm.cpp
)
add_compile_options(-lm)

target_link_libraries(swnlm 
    ${OpenCV_LIBS}
    swnlmLib
    utilsLib
)