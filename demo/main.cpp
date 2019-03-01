#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "Detector.h"
#include "util.h"
#include "Tracker.h"

using namespace std;
using namespace std::chrono;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        cerr << "usage: yolo-app <image path>" << endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Cannot open the video" << endl;
        return -2;
    }

    int64_t orig_dim[] = {cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH)};
    for (auto &x:orig_dim) {
        x = (x / 4 / (1 << 5) + 1) * (1 << 5);
    }
    Detector detector(orig_dim);
    Tracker tracker;

    cv::Mat origin_image;
    while (cap.read(origin_image)) {
        auto start = high_resolution_clock::now();
        auto detections = detector.detect(origin_image);

        auto bbox_acc = std::get<0>(detections).accessor<float, 2>();
        auto cls_acc = std::get<1>(detections).accessor<int64_t, 1>();

        vector<cv::Rect2f> dets;
        for (int64_t i = 0; i < cls_acc.size(0); ++i) {
            if (cls_acc[i] == 0) {
                auto x1 = bbox_acc[i][0];
                auto y1 = bbox_acc[i][1];
                auto x2 = bbox_acc[i][2];
                auto y2 = bbox_acc[i][3];
                dets.emplace_back(x1, y1, x2 - x1, y2 - y1);
            }
        }

        auto tracks = tracker.update(dets);

        auto end = high_resolution_clock::now();

        for (auto &t:tracks) {
            cv::rectangle(origin_image, t.box, cv::Scalar(0, 0, 255));
            draw_text(origin_image, to_string(t.id), cv::Scalar(0, 0, 255), t.box.tl());
        }

        draw_text(origin_image, "FPS: " + to_string(1000 / duration_cast<milliseconds>(end - start).count()),
                  {255, 255, 255}, cv::Point(origin_image.cols - 1, 0), true);
        cv::imshow("out", origin_image);
        if (cv::waitKey(1) != -1) break;
    }

    return 0;
}