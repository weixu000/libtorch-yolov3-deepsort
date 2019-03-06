#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include "KalmanTracker.h"

struct Track {
    int id;
    cv::Rect2f box;
};

class Tracker {
public:
    explicit Tracker(const std::array<int64_t, 2> &dim, int age = 10, int hits = 3, float iou = 0.3f)
            : img_box(0, 0, dim[1], dim[0]), max_age(age), min_hits(hits), iou_threshold(iou) {}

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets);

private:
    const cv::Rect2f img_box;
    const int max_age;
    const int min_hits;
    const float iou_threshold;

    std::vector<KalmanTracker> trackers;

    int frame_count = 0;
};

#endif //TRACKER_H
