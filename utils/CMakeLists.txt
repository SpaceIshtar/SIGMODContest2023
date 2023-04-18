file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(utils STATIC ${srcs})
target_include_directories(utils PUBLIC include)
#set_target_properties(utils PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dist_test test/dist_test.cpp)
target_link_libraries(dist_test utils gtest gtest_main)

add_executable(logger_test test/logger_test.cpp)
target_link_libraries(logger_test utils gtest gtest_main)

add_executable(io_test test/io_test.cpp)
target_link_libraries(io_test utils gtest gtest_main)

add_executable(sort_utils_test test/sort_utils_test.cpp)
target_link_libraries(sort_utils_test utils gtest gtest_main)

add_executable(thread_safe_test test/thread_safe_test.cpp)
target_link_libraries(thread_safe_test utils gtest gtest_main)

add_executable(rand_test test/random_uint_test.cpp)
target_link_libraries(rand_test utils gtest gtest_main)
target_link_options(rand_test PUBLIC -fopenmp)