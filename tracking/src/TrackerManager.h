#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <torch/torch.h>

#include "Track.h"
#include "KalmanTracker.h"

using DistanceMetricFunc = std::function<torch::Tensor()>;

class TrackerManager {
public:
    explicit TrackerManager(const std::array<int64_t, 2> &dim, float threshold)
            : img_box(0, 0, dim[1], dim[0]), dist_threshold(threshold) {}

    void predict();

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets,
                              const DistanceMetricFunc &metric);

    std::vector<KalmanTracker> trackers;
    std::vector<cv::Point> matched;

private:
    const cv::Rect2f img_box;
    const float dist_threshold;

    static const auto max_age = 10;
};

#endif //TRACKER_H
