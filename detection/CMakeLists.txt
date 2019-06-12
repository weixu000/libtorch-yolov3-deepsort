find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

aux_source_directory(src DETECTION_SRCS)
add_library(detection SHARED ${DETECTION_SRCS})

include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(detection)

target_link_libraries(detection PUBLIC ${OpenCV_LIBS} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(detection
        PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE src)