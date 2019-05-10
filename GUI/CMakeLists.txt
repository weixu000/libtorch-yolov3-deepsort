find_package(wxWidgets COMPONENTS core base REQUIRED)
include(${wxWidgets_USE_FILE})

aux_source_directory(. GUI_SRCS)

add_executable(GUI "${GUI_SRCS}")
target_link_libraries(GUI "${wxWidgets_LIBRARIES}" "${OpenCV_LIBS}" stdc++fs)
target_include_directories(GUI PRIVATE "${PROJECT_BINARY_DIR}")