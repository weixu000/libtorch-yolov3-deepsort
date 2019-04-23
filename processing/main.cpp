#include <iostream>
#include <opencv2/opencv.hpp>

#include "Detector.h"
#include "DeepSORT.h"
#include "TargetStorage.h"

using namespace std;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        cerr << "usage: processing <input path>" << endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Cannot open cv::VideoCapture" << endl;
        return -2;
    }

    array<int64_t, 2> orig_dim{int64_t(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(cap.get(cv::CAP_PROP_FRAME_WIDTH))};
    array<int64_t, 2> inp_dim;
    for (size_t i = 0; i < 2; ++i) {
        auto factor = 1 << 5;
        inp_dim[i] = (orig_dim[i] / 4 / factor + 1) * factor;
    }
    Detector detector(inp_dim);
    DeepSORT tracker(orig_dim);

    TargetStorage repo(orig_dim, static_cast<int>(cap.get(cv::CAP_PROP_FPS)));

    auto image = cv::Mat();
    while (cap.read(image)) {
        auto frame_processed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
        auto dets = detector.detect(image);
        auto trks = tracker.update(dets, image);

        repo.update(trks, frame_processed, image);

        cout << "Processed: " << frame_processed << endl;
    }
}