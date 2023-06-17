add_library(swnlmCudaLib "")   

target_sources(swnlmCudaLib
    PRIVATE 
        src/swnlmcuda.cu
        src/swilk.cuh
        #src/swilk.cu 
        src/swilkutils.cuh 
        src/swilkutils.cu
        
        src/helper_cuda.h
        src/helper_string.h
    PUBLIC 
        inc/swnlmcuda.cuh
)

target_include_directories(swnlmCudaLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(swnlmCudaLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)