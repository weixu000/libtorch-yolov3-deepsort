#ifndef SORT_H
#define SORT_H

#include <memory>

#include "Track.h"

class Tracker;

class SORT {
public:
    explicit SORT(const std::array<int64_t, 2> &dim);

    ~SORT();

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets);

private:
    std::unique_ptr<Tracker> tracker;
};

#endif //SORT_H
