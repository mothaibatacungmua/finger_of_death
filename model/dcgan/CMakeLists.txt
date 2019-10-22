cmake_minimum_required(VERSION 3.0)
project(dcgan)

find_package(Torch REQUIRED)
set(SOURCE dcgan.cpp)
add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)