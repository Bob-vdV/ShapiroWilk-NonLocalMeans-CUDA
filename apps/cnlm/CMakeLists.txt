add_compile_options(-lm -lstdc++fs)

add_executable(cnlm
    cnlm.cpp
)

target_include_directories( cnlm PRIVATE ${ROOT_SOURCE_DIR}/images)

target_link_libraries(cnlm 
    ${OpenCV_LIBS}
    cnlmLib
    utilsLib
)
