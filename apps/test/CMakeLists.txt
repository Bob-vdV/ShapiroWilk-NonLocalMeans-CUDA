add_executable(test
	main.cpp
)

target_include_directories( test PRIVATE ${ROOT_SOURCE_DIR}/images)

target_link_libraries( test 
	${OpenCV_LIBS}
)
