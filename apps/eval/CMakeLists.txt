add_compile_options(-lm -lstdc++fs)

add_executable(eval
    eval.cpp
)

target_include_directories( eval PRIVATE ${ROOT_SOURCE_DIR}/images)

target_link_libraries(eval 
    ${OpenCV_LIBS}
    cnlmLib
    swnlmLib
    utilsLib
)
