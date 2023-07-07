add_library(cnlmLib "")   

target_sources(cnlmLib
    PRIVATE 
        src/cnlm.cpp
    PUBLIC 
        inc/cnlm.hpp
)

target_include_directories(cnlmLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(cnlmLib
    commonLib
    ${OpenCV_LIBS}
)