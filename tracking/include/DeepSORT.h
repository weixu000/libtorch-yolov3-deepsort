#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "Track.h"

class Extractor;

class TrackerManager;

class FeatureMetric;

class DeepSORT {
public:
    explicit DeepSORT(const std::array<int64_t, 2> &dim);

    ~DeepSORT();

    std::vector<Track> update(const std::vector<cv::Rect2f> &dets, cv::Mat ori_img);

private:
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<TrackerManager> manager;
    std::unique_ptr<FeatureMetric> feat_metric;
};


#endif //DEEPSORT_H
