cmake_minimum_required(VERSION 3.0)
project(bert)

include_directories(${CMAKE_SOURCE_DIR}/third_party)
file(GLOB SRCS *.cpp)
set(LIB_SOURCES bert_config.cpp modeling.cpp utils.cpp common.cpp)

add_executable(${PROJECT_NAME} ${SRCS})
add_library(lib_bert ${LIB_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/bert"
)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(lib_bert "${TORCH_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lib_bert PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
