add_library(swnlmLib "")   

target_sources(swnlmLib
    PRIVATE 
        src/noise.hpp 
        src/noise.cpp 
        src/swilkutils.hpp
        src/swilkutils.cpp
        src/swilk.hpp
        src/swilk.cpp
        src/swnlm.cpp
    PUBLIC 
        inc/swnlm.hpp
)

target_include_directories(swnlmLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(swnlmLib
    ${OpenCV_LIBS}
)