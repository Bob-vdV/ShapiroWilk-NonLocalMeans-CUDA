add_executable(hdnlm
	main.cpp 
	kmeans.hpp 
	kmeans.cpp 
	fastapprox.hpp
	fastapprox.cpp
	filter.hpp
	filter.cpp
	)
add_compile_options(-lm)

target_link_libraries(hdnlm ${OpenCV_LIBS} )