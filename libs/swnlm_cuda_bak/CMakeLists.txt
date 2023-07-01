add_library(swnlmCudaLib "")   

target_sources(swnlmCudaLib
    PRIVATE 
        src/swnlmcuda.cu
        src/swilk.cuh
        src/swilkutils.cuh 
        src/bitonicsort.cuh
        src/sort.cuh
        src/radixsort.cuh
        src/countingsort.cuh

    PUBLIC 
        inc/swnlmcuda.cuh
)

target_include_directories(swnlmCudaLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(swnlmCudaLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)