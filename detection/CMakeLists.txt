aux_source_directory(src DETECTION_SRCS)
add_library(detection ${DETECTION_SRCS})
target_link_libraries(detection "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
target_include_directories(detection
        PUBLIC include
        PRIVATE src)