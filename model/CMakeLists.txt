cmake_minimum_required(VERSION 3.0)
project(finger_of_death)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(dcgan)
add_subdirectory(bert)