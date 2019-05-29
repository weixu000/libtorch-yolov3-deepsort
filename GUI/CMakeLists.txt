find_package(wxWidgets COMPONENTS core base REQUIRED)
include(${wxWidgets_USE_FILE})

aux_source_directory(. GUI_SRCS)

add_executable(GUI WIN32 "${GUI_SRCS}")
target_link_libraries(GUI ${wxWidgets_LIBRARIES} ${OpenCV_LIBS} ${STDCXXFS})
target_include_directories(GUI PRIVATE "${PROJECT_BINARY_DIR}")
