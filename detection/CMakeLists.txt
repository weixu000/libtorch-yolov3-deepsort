aux_source_directory(src DETECTION_SRCS)
add_library(detection SHARED ${DETECTION_SRCS})
target_link_libraries(detection PUBLIC "${OpenCV_LIBS}" PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(detection
        PUBLIC include
        PRIVATE src)