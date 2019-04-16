#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include <torch/torch.h>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

torch::Tensor iou_dist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks);

//A tool to calculate distance;
class FeatureMetric {
public:
    torch::Tensor distance(torch::Tensor features, const std::vector<int> &targets);

    void update(torch::Tensor feats, const std::vector<int> &active_targets, const std::vector<int> &remain_targets);

private:
    class FeatureBundle;

    std::map<int, FeatureBundle> samples;
};

class FeatureMetric::FeatureBundle {
public:
    FeatureBundle() : full(false), next(0), store(torch::empty({budget, feat_dim})) {}

    void add(torch::Tensor feat);

    torch::Tensor get();

private:
    static const int64_t budget = 100, feat_dim = 512;

    torch::Tensor store;

    bool full;
    int64_t next;
};

#endif //NN_MATCHING_H
