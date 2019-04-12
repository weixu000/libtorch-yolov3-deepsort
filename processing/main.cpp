#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

#include "Detector.h"
#include "SORT.h"
#include "TargetStorage.h"

using namespace std;
namespace fs = std::experimental::filesystem;

const string output_dir = "result";

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

    auto video_FPS = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    array<int64_t, 2> orig_dim{int64_t(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(cap.get(cv::CAP_PROP_FRAME_WIDTH))};

    fs::create_directories(output_dir);

    cv::VideoWriter writer((fs::path(output_dir) / "compressed.flv").string(),
                           cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                           video_FPS, cv::Size(orig_dim[1], orig_dim[0]));
    if (!writer.isOpened()) {
        cerr << "Cannot open cv::VideoWriter" << endl;
        return -3;
    }

    array<int64_t, 2> inp_dim;
    for (size_t i = 0; i < 2; ++i) {
        auto factor = 1 << 5;
        inp_dim[i] = (orig_dim[i] / 3 / factor + 1) * factor;
    }
    Detector detector(inp_dim);
    SORT tracker(orig_dim);

    TargetStorage repo(output_dir);

    auto image = cv::Mat();
    while (cap.read(image)) {
        auto frame_processed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
        repo.update(tracker.update(detector.detect(image)), frame_processed, image);

        writer.write(image);

        repo.record();
        cout << "Processed: " << frame_processed << endl;
    }
}