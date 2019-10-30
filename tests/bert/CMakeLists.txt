file(GLOB SRCS *.cpp)

add_executable(bert_test ${SRCS})

target_link_directories(bert_test PUBLIC
    bert
    gtest
    gmock
    gtest_main
)

add_test(NAME bert_test
         COMMAND bert_test)

add_custom_command(TARGET bert_test PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_SOURCE_DIR}/tests/bert/data/ $<TARGET_FILE_DIR:bert_test>/data/)