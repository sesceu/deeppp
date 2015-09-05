cmake_minimum_required(VERSION 2.8)

project(deeppp)

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -W")

set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
if(NOT EIGEN3_INCLUDE_DIR)
	message(FATAL_ERROR "Cannot find Eigen3 include directory.")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})

set(LIBRARY_SOURCES
        src/mnistloader.cpp
        src/random.cpp
)

set(LIBRARY_HEADERS
	src/datacoder.h
	src/mnistimage.h
	src/mnistloader.h
        src/layer.h
	src/restrictedboltzmannlayer.h
	src/timeserieslayer.h
	src/random.h
)

add_library(deeppp ${LIBRARY_SOURCES} )
target_include_directories(deeppp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC ${EIGEN3_INCLUDE_DIR}
)

#install(FILES ${LIBRARY_HEADERS} DESTINATION "deeppp" PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
