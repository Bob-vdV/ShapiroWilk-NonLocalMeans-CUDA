add_library(swnlmLib "")   

target_sources(swnlmLib
    PRIVATE 
        src/swnlm.cpp
    PUBLIC 
        inc/swnlm.hpp
)

target_include_directories(swnlmLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(swnlmLib
    swilkLib
    ${OpenCV_LIBS}
)