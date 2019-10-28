cmake_minimum_required(VERSION 3.0)
project(dcgan)

set(SOURCE dcgan.cpp)

add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
