cmake_minimum_required(VERSION 3.0)
project(bert)

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Torch REQUIRED)
add_executable(${PROJECT_NAME} main.cpp bert_config.cpp utils.cpp modeling.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)