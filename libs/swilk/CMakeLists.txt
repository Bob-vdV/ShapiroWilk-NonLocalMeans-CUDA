add_library(swilkLib )   

set_target_properties(swilkLib PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    )

target_sources(swilkLib
    PRIVATE 
        src/swilk.cu
        src/sort.cu
        src/sort.cuh

    PUBLIC 
        inc/swilk.cuh
)

target_include_directories(swilkLib  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
