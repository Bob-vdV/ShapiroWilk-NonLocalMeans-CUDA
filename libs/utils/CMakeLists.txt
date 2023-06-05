add_library(utilsLib "")   

target_sources(utilsLib
    PRIVATE 
        src/utils.cpp
    PUBLIC 
        inc/utils.hpp
)

target_include_directories(utilsLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)

target_link_libraries( utilsLib 
    ${OpenCV_LIBS}
)
