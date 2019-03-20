#ifndef TARGET_H
#define TARGET_H

#include <map>
#include <utility>
#include <opencv2/opencv.hpp>

#include "Tracker.h"
#include "util.h"

class TargetStorage {
public:
    void update(const std::vector<Track> &trks,
                int frame, const cv::Mat &image);

    void record();

private:
    struct Target {
        std::map<int, cv::Rect2f> trajectories;
        std::map<int, cv::Mat> snapshots;
        int last_snap = 0;
    };

    static constexpr float padding = 0.1f;

    std::map<int, Target> targets;
};

#endif //TARGET_H
