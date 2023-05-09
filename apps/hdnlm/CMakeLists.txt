add_executable(hdnlm
	main.cpp
)

add_compile_options(-lm)

target_link_libraries(hdnlm ${OpenCV_LIBS} )