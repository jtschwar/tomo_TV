
cmake_minimum_required(VERSION 3.1)
project( tomo_tv ) 


find_package( OpenCV )
include_directories( ./Dependencies/Eigen/ )


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -O3 -Wno-div-by-zero")


############ ASD TV #################
add_executable( asd_tv asd_tv.cxx )
target_link_libraries( asd_tv ${OpenCV_LIBS} ) 
include_directories( asd_tv ${OpenCV_INCLUDE_DIRS} )


############ Dynamic TV #################
add_executable( dynamic_tv dynamic_tv.cxx )
target_link_libraries( dynamic_tv ${OpenCV_LIBS} )
include_directories( dynamic_tv ${OpenCV_INCLUDE_DIRS} )

############ Time TV #################
add_executable( time_tv time_tv.cxx )
target_link_libraries( time_tv ${OpenCV_LIBS} )
include_directories( time_tv ${OpenCV_INCLUDE_DIRS} )


############ ART #################
#add_executable( art art.cxx ) 
#target_link_libraries( art ${OpenCV_LIBS} ) 
#include_directories( art ${OpenCV_INCLUDE_DIRS} )


##### Projected Line Search ########
#add_executable( line_tv proj_line_tv.cxx )
#target_link_libraries( line_tv ${OpenCV_LIBS} ) 
#include_directories( line_tv ${OpenCV_INCLUDE_DIRS} ) 