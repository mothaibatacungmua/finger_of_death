file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(bert_test ${SRCS})

TARGET_LINK_LIBRARIES(bert_test
    bert
    gtest_main
)

add_test(NAME bert_test
         COMMAND bert_test)