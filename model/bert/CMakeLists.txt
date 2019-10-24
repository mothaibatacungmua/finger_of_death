cmake_minimum_required(VERSION 3.0)
project(bert)

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bert main.cpp bert_config.cpp)
