cmake_minimum_required(VERSION 3.13)
project( NonLocalMeans VERSION 1.00)

set (CMAKE_CXX_STANDARD_REQUIRED 17)
set (CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_compile_options(-Wall)

add_subdirectory(libs)
add_subdirectory(apps)