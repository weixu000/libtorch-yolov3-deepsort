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
        cerr << "usage: yolo-app <image path>" << endl;
        return -1;
    }

    Detector detector;

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Cannot open the video" << endl;
        return -2;
    }

    cv::namedWindow("out");
    cv::Mat origin_image;

    while (cap.read(origin_image)) {
        auto start = high_resolution_clock::now();
        auto detections = detector.detect(origin_image);
        auto end = high_resolution_clock::now();

        draw_detections(origin_image, detections, classes, cmap);
        draw_text(origin_image, "FPS: " + to_string(1000 / duration_cast<milliseconds>(end - start).count()),
                  cv::Scalar(255, 255, 255), cv::Point(origin_image.cols - 1, 0), true);
        cv::imshow("out", origin_image);
        if (cv::waitKey(1) != -1) break;
    }

    return 0;
}