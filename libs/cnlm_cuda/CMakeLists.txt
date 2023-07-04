add_library(cnlmCudaLib "")   

target_sources(cnlmCudaLib
    PRIVATE 
        src/cnlmcuda.cu

    PUBLIC 
        inc/cnlmcuda.cuh
)

target_include_directories(cnlmCudaLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(cnlmCudaLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)