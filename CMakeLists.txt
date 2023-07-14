cmake_minimum_required(VERSION 3.18)
project( NonLocalMeans VERSION 1.00 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


set (CMAKE_CXX_STANDARD_REQUIRED 17)
set (CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_package(CUDAToolkit)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -rdc=true")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    #compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(libs)
add_subdirectory(apps)