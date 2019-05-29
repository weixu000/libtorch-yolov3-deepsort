#ifndef SORT_H
#define SORT_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

#include "tracking_export.h"
#include "Track.h"

template<typename T>
class TrackerManager;

class TRACKING_EXPORT  SORT {
public:
    explicit SORT(const std::array<int64_t, 2> &dim);

    ~SORT();

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets);

private:
    class TrackData;

    std::vector<TrackData> data;

    std::unique_ptr<TrackerManager<TrackData>> manager;
};

#endif //SORT_H
