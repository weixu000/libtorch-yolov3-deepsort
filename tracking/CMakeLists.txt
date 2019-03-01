aux_source_directory(src TRACKING_SRCS)
add_library(tracking ${TRACKING_SRCS})
target_link_libraries(tracking "${OpenCV_LIBS}")
target_include_directories(tracking
        PUBLIC include
        PRIVATE src)