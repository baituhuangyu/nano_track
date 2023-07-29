cmake_minimum_required(VERSION 3.16)
project(opencv_nano_track)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall  -Wextra -g ")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -lfftw3 -lm")

find_package(OpenCV 4.7 PATHS "/home/hy/github/opencv/opencv/cuda_release/opencvlib/")
#find_package(OpenCV 4)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
include_directories(. ${OpenCV_INCLUDE_DIRS} )

#FIND_PACKAGE(Eigen REQUIRED)

add_executable(opencv_nano_track
        opencv_dnn/main_opencv_dnn.cpp
        opencv_dnn/tracker_nano2.cpp)
target_link_libraries(opencv_nano_track
        ${OpenCV_LIBS}
        fftw3
        m
        pthread
        )