cmake_minimum_required(VERSION 3.0)
project(dcgan)

set(SOURCE dcgan.cpp)

add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/dcgan"
)

get_property(TARGET_LOC TARGET dcgan PROPERTY RUNTIME_OUTPUT_DIRECTORY)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/display_images.py
     DESTINATION ${TARGET_LOC})
file(COPY ${CMAKE_SOURCE_DIR}/etc/download_mnist.py
     DESTINATION ${TARGET_LOC})
     