#ifndef TARGET_H
#define TARGET_H

#include <map>
#include <array>
#include <utility>
#include <opencv2/opencv.hpp>

#include "Track.h"
#include "util.h"

class TargetStorage {
public:
    explicit TargetStorage(const std::array<int64_t, 2> &orig_dim, int video_FPS);

    virtual ~TargetStorage() { record(0); }

    void update(const std::vector<Track> &trks,
                int frame, const cv::Mat &image);

    struct Target {
        std::map<int, cv::Rect2f> trajectories;
        std::map<int, cv::Mat> snapshots;
        int last_snap = 0;
    };

    const std::map<int, Target> &get() const { return targets; }

private:
    void record(int remain);

    static constexpr float padding = 0.1f;

    std::map<int, Target> targets;

    cv::VideoWriter writer;
};

#endif //TARGET_H
