cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 61)

project( NonLocalMeans VERSION 1.00 LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD_REQUIRED 17)
set (CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    #compile_options(add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(libs)
add_subdirectory(apps)