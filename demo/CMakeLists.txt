aux_source_directory(. DEMO_SRCS)
add_executable(demo ${DEMO_SRCS})
target_link_libraries(demo "${OpenCV_LIBS}" detection tracking)