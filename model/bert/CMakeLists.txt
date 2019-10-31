cmake_minimum_required(VERSION 3.0)
project(bert)

include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SRCS *.cpp)

add_executable(${PROJECT_NAME} ${SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
  ENABLE_EXPORTS 1
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/bert"
)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
