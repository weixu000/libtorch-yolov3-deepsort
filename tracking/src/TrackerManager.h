#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <torch/torch.h>
#include <tuple>

#include "Track.h"
#include "KalmanTracker.h"

using DistanceMetricFunc = std::function<
        torch::Tensor(const std::vector<int> &trk_ids, const std::vector<int> &det_ids)>;

class TrackerManager {
public:
    explicit TrackerManager(const std::array<int64_t, 2> &dim)
            : img_box(0, 0, dim[1], dim[0]) {}

    std::vector<int> predict();

    std::tuple<std::vector<std::tuple<int, int>>, std::vector<int>>
    update(const std::vector<cv::Rect2f> &dets,
           const DistanceMetricFunc &confirmed_metric, const DistanceMetricFunc &unconfirmed_metric);

    std::vector<Track> visible_tracks();

    std::vector<KalmanTracker> trackers;

    static const float invalid_dist;

private:
    const cv::Rect2f img_box;
};

#endif //TRACKER_H
