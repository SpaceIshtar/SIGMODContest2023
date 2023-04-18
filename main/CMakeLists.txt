add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent efanna)
target_link_options(test_nndescent PUBLIC -O3 -fopenmp )


# add_executable(test_nndescent_batch test_nndescent_batch.cpp)
# target_link_libraries(test_nndescent_batch efanna)
# target_link_options(test_nndescent_batch PUBLIC -O3 -fopenmp)
# target_link_options(test_nndescent_batch PUBLIC -fopenmp -g)

# add_executable(test_filter test_filter.cpp)
# target_link_libraries(test_filter cuckoofilter)
# target_link_options(test_filter PUBLIC -O3)



# add_executable(sample_data sample_data.cpp)
# target_link_libraries(sample_data utils)
# target_link_options(sample_data PUBLIC -O3 -fopenmp)


