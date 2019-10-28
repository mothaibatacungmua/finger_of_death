cmake_minimum_required(VERSION 3.0)
project(finger_of_death)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(Torch REQUIRED)
add_subdirectory(model)

include(cmake-gtest/googletest.cmake)
fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake-gtest
    ${PROJECT_BINARY_DIR}/googletest
)

enable_testing()
add_subdirectory(tests)