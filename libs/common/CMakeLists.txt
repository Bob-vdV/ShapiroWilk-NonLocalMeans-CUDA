add_library(commonLib "")   

target_compile_options(commonLib PUBLIC -lm)

target_sources(commonLib
    PRIVATE 
        src/common.cpp
    PUBLIC 
        inc/common.hpp
)

target_include_directories(commonLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)

#target_link_libraries( commonLib 
#    ${OpenCV_LIBS}
#)
