find_package(glfw3 REQUIRED)

aux_source_directory(. IMGUI_SRC)

add_library(imgui ${IMGUI_SRC})
target_link_libraries(imgui glfw gl3w)
target_include_directories(imgui PUBLIC .)
