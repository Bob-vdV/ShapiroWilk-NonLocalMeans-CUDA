add_executable(hdnlm
	main.cpp 
	pca.hpp
	pca.cpp
	kmeans.hpp 
	kmeans.cpp 
	fastapprox.hpp
	fastapprox.cpp
	filter.hpp
	filter.cpp
	utils.hpp
	utils.cpp
	utils.tpp
	)
add_compile_options(-lm)

target_link_libraries(hdnlm ${OpenCV_LIBS} )