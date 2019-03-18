find_package(OpenGL REQUIRED)

aux_source_directory(GL GL3W_SRC)

add_library(gl3w ${GL3W_SRC})
target_link_libraries(gl3w ${CMAKE_DL_LIBS} OpenGL::GL)
target_include_directories(gl3w PUBLIC .)
