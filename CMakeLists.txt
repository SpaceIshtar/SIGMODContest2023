cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "FLAGS = ${CMAKE_CXX_FLAGS}")


set(CMAKE_EXPORT_COMPILECOMMANDS ON)

project(SIGMOD_Contest LANGUAGES CXX)

message(STATUS "ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "FLAGS = ${CMAKE_CXX_FLAGS}")

add_compile_options(-O3  -lrt -std=c++17 -march=native -fPIC -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0)

add_subdirectory(efanna)
add_subdirectory(main)