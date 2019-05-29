#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "tracking_export.h"
#include "Track.h"

class Extractor;

template<typename T>
class TrackerManager;

template<typename T>
class FeatureMetric;

class TRACKING_EXPORT DeepSORT {
public:
    explicit DeepSORT(const std::array<int64_t, 2> &dim);

    ~DeepSORT();

    std::vector<Track> update(const std::vector<cv::Rect2f> &detections, cv::Mat ori_img);

private:
    class TrackData;

    std::vector<TrackData> data;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<TrackerManager<TrackData>> manager;
    std::unique_ptr<FeatureMetric<TrackData>> feat_metric;
};


#endif //DEEPSORT_H
