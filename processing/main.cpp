#include <iostream>
#include <opencv2/opencv.hpp>

#include "Detector.h"
#include "Tracker.h"
#include "TargetStorage.h"

using namespace std;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        cerr << "usage: processing <image path>" << endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Cannot open cv::VideoCapture" << endl;
        return -2;
    }

    auto video_FPS = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    array<int64_t, 2> orig_dim{cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH)};

    array<int64_t, 2> inp_dim;
    for (size_t i = 0; i < 2; ++i) {
        auto factor = 1 << 5;
        inp_dim[i] = (orig_dim[i] / factor + 1) * factor;
    }
    Detector detector(inp_dim);
    Tracker tracker(orig_dim);

    TargetStorage repo;

    auto image = cv::Mat();
    while (cap.read(image)) {
        auto frame_processed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
        repo.update(tracker.update(detector.detect(image)), frame_processed, image);

        if (frame_processed % 20 == 0) {
            repo.record();
        }
        cout << "Processed: " << frame_processed << endl;
    }
    repo.record();
}