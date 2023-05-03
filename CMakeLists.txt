cmake_minimum_required(VERSION 3.0)
project( NonLocalMeans VERSION 1.00)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( serial main.cpp )
add_executable( test test.cpp )

target_link_libraries( serial ${OpenCV_LIBS} )
target_link_libraries( test ${OpenCV_LIBS} )