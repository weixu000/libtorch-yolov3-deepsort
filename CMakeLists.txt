cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(MyProject)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(detection)
add_subdirectory(tracking)
add_subdirectory(demo)
