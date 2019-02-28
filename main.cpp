#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "Detector.h"
#include "util.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[]) {
    auto classes = load_classes("data/coco.names");
    auto cmap = color_map(classes.size());

    if (argc != 2) {
        std::cerr << "usage: yolo-app <image path>" << std::endl;
        return -1;
    }

    Detector detector;

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open the video" << std::endl;
        return -2;
    }

    cv::namedWindow("out");
    cv::Mat origin_image;

    std::cout << "start to inference ..." << endl;

    while (cap.read(origin_image)) {
        auto start = std::chrono::high_resolution_clock::now();
        auto detections = detector.detect(origin_image);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "FPS: " << 1000 / duration_cast<milliseconds>(end - start).count() << endl;

        draw_detections(origin_image, detections, classes, cmap);
        cv::imshow("out", origin_image);
        if (cv::waitKey(1) != -1) break;
    }

    std::cout << "Done" << endl;

    return 0;
}