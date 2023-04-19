add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent efanna)
target_link_options(test_nndescent PUBLIC -O3 -fopenmp )