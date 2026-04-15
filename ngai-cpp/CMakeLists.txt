cmake_minimum_required(VERSION 3.16)
project(ngai-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Main inference engine
add_executable(ngai
    src/main.cpp
    src/tensor.cpp
    src/ternary.cpp
    src/model.cpp
)

target_include_directories(ngai PRIVATE include)

# Tests
add_executable(ngai_test
    tests/test_ternary.cpp
    src/tensor.cpp
    src/ternary.cpp
)

target_include_directories(ngai_test PRIVATE include)
