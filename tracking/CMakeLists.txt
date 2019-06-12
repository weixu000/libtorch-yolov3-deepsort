find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

aux_source_directory(src TRACKING_SRCS)
add_library(tracking SHARED ${TRACKING_SRCS})

include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(tracking)

target_link_libraries(tracking PUBLIC ${OpenCV_LIBS} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(tracking
        PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE src)