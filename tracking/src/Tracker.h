#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <torch/torch.h>

#include "Track.h"
#include "KalmanTracker.h"

using DistanceMetricFunc = std::function<torch::Tensor()>;

class Tracker {
public:
    explicit Tracker(const std::array<int64_t, 2> &dim)
            : img_box(0, 0, dim[1], dim[0]) {}

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets,
                              const DistanceMetricFunc &metric);

    std::vector<KalmanTracker> trackers;

private:
    const cv::Rect2f img_box;
    static const auto max_age = 10;
    static const auto min_hits = 3;

    int frame_count = 0;
};

#endif //TRACKER_H
