aux_source_directory(. PROCESSING_SRCS)

add_executable(processing ${PROCESSING_SRCS})
target_link_libraries(processing "${OpenCV_LIBS}" detection tracking stdc++fs)