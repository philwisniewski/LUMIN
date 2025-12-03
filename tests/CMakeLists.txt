cmake_minimum_required(VERSION 3.16)

enable_testing()

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

add_executable(test_matrix test_matrix.cpp)

target_link_libraries(test_matrix
    lumin
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

gtest_discover_tests(test_matrix)

