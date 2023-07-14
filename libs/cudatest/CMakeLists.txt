add_library(cudaTest "")   

target_sources(cudaTest
    PRIVATE 
        cudatest.cu
    PUBLIC 
        inc/cudatest.cuh
)

target_include_directories(cudaTest PUBLIC ${CMAKE_CURRENT_LIST_DIR}/inc)
#target_link_libraries(cudaTest
#    ${}
#    ${OpenCV_LIBS}
#)