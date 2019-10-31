cmake_minimum_required(VERSION 3.5)
project(finger_of_death)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_package(Torch REQUIRED)
add_subdirectory(model)

include(cmake-gtest/googletest.cmake)
fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake-gtest
    ${PROJECT_BINARY_DIR}/googletest
)
# include_directories(${CMAKE_BINARY_DIR}/googletest/googletest-src/googletest/include)
# include_directories(${CMAKE_BINARY_DIR}/googletest/googletest-src/googlemock/include)

enable_testing()
add_subdirectory(tests)