#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include "KalmanTracker.h"

struct Track {
    int id;
    Rect2f box;
};

class Tracker {
public:
    explicit Tracker(int max_age = 10, int min_hits = 3)
            : max_age(max_age), min_hits(min_hits) {}

    std::vector<Track> update(const std::vector<Rect2f> &dets);

private:
    const int max_age;
    const int min_hits;
    const float iouThreshold = 0.3;

    std::vector<KalmanTracker> trackers;

    int frame_count = 0;
};


#endif //TRACKER_H
