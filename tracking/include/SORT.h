#ifndef SORT_H
#define SORT_H

#include <opencv2/opencv.hpp>
#include <memory>

#include "Track.h"

class TrackerManager;

class SORT {
public:
    explicit SORT(const std::array<int64_t, 2> &dim);

    ~SORT();

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets);

private:
    std::unique_ptr<TrackerManager> manager;
};

#endif //SORT_H
