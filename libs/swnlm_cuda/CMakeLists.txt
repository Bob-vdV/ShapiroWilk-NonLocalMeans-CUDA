add_library(swnlmCudaLib "")   

target_sources(swnlmCudaLib
    PRIVATE 
        src/swnlmcuda.cu
    PUBLIC 
        inc/swnlmcuda.cuh
)

target_include_directories(swnlmCudaLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
target_link_libraries(swnlmCudaLib
    swilkLib
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)