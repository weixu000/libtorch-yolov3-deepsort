add_subdirectory(gl3w)
add_subdirectory(imgui)

find_package(Threads REQUIRED)

aux_source_directory(. GUI_SRCS)

add_executable(GUI ${GUI_SRCS})
target_link_libraries(GUI "${OpenCV_LIBS}" gl3w imgui stdc++fs Threads::Threads)
target_include_directories(GUI PRIVATE "${PROJECT_BINARY_DIR}")