cmake_minimum_required(VERSION 3.22)

# Project name and version
project(MCExample VERSION 1.0
        DESCRIPTION "Simple example of using MPI"
        LANGUAGES CXX)

# Add the executable
add_executable(mc src/main.cpp)

# Specify the C++ standard
target_compile_features(mc PUBLIC cxx_std_11)

# Find MPI
find_package(MPI REQUIRED)
target_link_libraries(mc PUBLIC MPI::MPI_CXX)

