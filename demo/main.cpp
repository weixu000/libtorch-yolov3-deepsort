#include <opencv2/opencv.hpp>

#include "Detector.h"
#include "util.h"
#include "Tracker.h"

using namespace std;
using namespace std::chrono;

struct Target {
    vector<pair<int, cv::Rect2f>> trajectories;
};

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
        auto factor = 1 << 5;
        x = (x / 3 / factor + 1) * factor;
    }
    Detector detector(orig_dim);
    Tracker tracker;

    std::map<int, Target> targets;

    cv::Mat origin_image;
    for (int frame = 0; cap.read(origin_image); ++frame) {
        auto start = high_resolution_clock::now();
        auto dets = detector.detect(origin_image);
        auto tracks = tracker.update(dets);
        auto end = high_resolution_clock::now();

        for (auto &t:tracks) {
            targets[t.id].trajectories.emplace_back(frame, t.box);
        }

        for (auto &[id, t]:targets) {
            if (t.trajectories.back().first == frame) {
                cv::rectangle(origin_image, t.trajectories.back().second, {0, 0, 255});
                draw_text(origin_image, to_string(id), {0, 0, 255}, t.trajectories.back().second.tl());

                for (auto it = t.trajectories.begin(); it + 1 != t.trajectories.end(); ++it) {
                    auto pt1 = (it->second.tl() + it->second.br()) / 2;
                    auto pt2 = ((it + 1)->second.tl() + (it + 1)->second.br()) / 2;
                    cv::line(origin_image, pt1, pt2, {0, 0, 255});
                }
            }
        }

        draw_text(origin_image, "FPS: " + to_string(1000 / duration_cast<milliseconds>(end - start).count()),
                  {255, 255, 255}, cv::Point(origin_image.cols - 1, 0), true);
        cv::imshow("out", origin_image);

        auto c = char(cv::waitKey(1));
        if (c == ' ') {
            cv::waitKey(0);
        } else if (c == 'q') {
            break;
        }
//        cv::waitKey(0);
    }

    return 0;
}