#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "Detector.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[]) {
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

        auto result = detector.detect(origin_image);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "FPS: " << 1000 / duration_cast<milliseconds>(end - start).count() << endl;

        auto bbox = std::get<0>(result);
        auto classes = std::get<1>(result);
        auto scores = std::get<2>(result);

        std::cout << bbox.size(0) << " objects found" << endl;

        auto result_data = bbox.accessor<float, 2>();

        for (int i = 0; i < bbox.size(0); i++) {
            cv::rectangle(origin_image, cv::Point(result_data[i][0], result_data[i][1]),
                          cv::Point(result_data[i][2], result_data[i][3]), cv::Scalar(0, 0, 255), 1, 1, 0);
        }
        cv::imshow("out", origin_image);

        if (cv::waitKey(1) != -1) break;
    }

    std::cout << "Done" << endl;

    return 0;
}