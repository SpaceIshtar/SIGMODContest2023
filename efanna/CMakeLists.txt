file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp src/*.cc include/*.h include/*.hpp)

add_library(efanna STATIC ${srcs})
target_include_directories(efanna PUBLIC include)
target_compile_options(efanna PUBLIC -O3 -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)
