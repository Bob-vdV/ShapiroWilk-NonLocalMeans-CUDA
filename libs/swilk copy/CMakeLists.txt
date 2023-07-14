add_library(swilkLib )   

set_target_properties(swilkLib PROPERTIES 
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    )

target_sources(swilkLib
    PRIVATE 
        #src/swilk.cu
        #src/sort.cuh
        src/test.cu
        src/test.cuh 
    PUBLIC 
        inc/swilk.cuh
)

target_include_directories(swilkLib  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(swilkLib
    ${CUDA_LIBRARIES}
)