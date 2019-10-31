file(GLOB SRCS *.cpp)
project(bert_test)

add_executable(${PROJECT_NAME} ${SRCS})

target_link_libraries(${PROJECT_NAME}
    bert
    gtest_main
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  ENABLE_EXPORTS 1
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/bert/tests"
)

add_test(NAME ${PROJECT_NAME}
         COMMAND ${PROJECT_NAME})

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_SOURCE_DIR}/tests/bert/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/tests/data/)
