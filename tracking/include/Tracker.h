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
    std::vector<Track> update(const std::vector<cv::Rect2f> &dets);

private:
    static const int max_age;
    static const int min_hits;
    static const float iou_threshold;

    std::vector<KalmanTracker> trackers;

    int frame_count = 0;
};

#endif //TRACKER_H
