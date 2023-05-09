cmake_minimum_required(VERSION 3.0)
project( NonLocalMeans VERSION 1.00)

#set (CMAKE_CXX_STANDARD_REQUIRED 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_compile_options(-Wall)

add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/images
    ${PROJECT_BINARY_DIR}/images
    COMMENT "Copying images into binary directory"
)

add_subdirectory(apps)