find_package(OpenCV REQUIRED)
find_package(wxWidgets COMPONENTS core base REQUIRED)

include(${wxWidgets_USE_FILE})

aux_source_directory(. GUI_SRCS)

if (WIN32)
    set(WIN_RC "win.rc")
else ()
    set(WIN_RC "")
endif ()

add_executable(GUI WIN32 "${GUI_SRCS}" ${WIN_RC})
target_link_libraries(GUI ${wxWidgets_LIBRARIES} ${OpenCV_LIBS} ${STDCXXFS})
target_include_directories(GUI PRIVATE "${PROJECT_BINARY_DIR}")
