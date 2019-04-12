aux_source_directory(src TRACKING_SRCS)
add_library(tracking SHARED ${TRACKING_SRCS})
target_link_libraries(tracking PUBLIC "${OpenCV_LIBS}" PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(tracking
        PUBLIC include
        PRIVATE src)