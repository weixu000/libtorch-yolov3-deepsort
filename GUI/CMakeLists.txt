add_subdirectory(gl3w)
add_subdirectory(imgui)

aux_source_directory(. GUI_SRCS)

add_executable(GUI ${GUI_SRCS})
target_link_libraries(GUI "${OpenCV_LIBS}" gl3w imgui stdc++fs)