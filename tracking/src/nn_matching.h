#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include <torch/torch.h>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

torch::Tensor iou_dist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks);

// save features of the track in GPU
class FeatureBundle {
public:
    FeatureBundle() : full(false), next(0), store(torch::empty({budget, feat_dim}).cuda()) {}

    void clear() {
        next = 0;
        full = false;
    }

    bool empty() const {
        return next == 0 && !full;
    }

    void add(torch::Tensor feat) {
        if (next == budget) {
            full = true;
            next = 0;
        }
        store[next++] = feat;
    }

    torch::Tensor get() const {
        return full ? store : store.slice(0, 0, next);
    }

private:
    static const int64_t budget = 100, feat_dim = 512;

    torch::Tensor store;

    bool full;
    int64_t next;
};

template<typename TrackData>
class FeatureMetric {
public:
    explicit FeatureMetric(std::vector<TrackData> &data) : data(data) {}

    torch::Tensor distance(torch::Tensor features, const std::vector<int> &targets) {
        auto dist = torch::empty({int64_t(targets.size()), features.size(0)});
        if (features.size(0)) {
            for (size_t i = 0; i < targets.size(); ++i) {
                dist[i] = nn_cosine_distance(data[targets[i]].feats.get(), features);
            }
        }

        return dist;
    }

    void update(torch::Tensor feats, const std::vector<int> &targets) {
        for (size_t i = 0; i < targets.size(); ++i) {
            data[targets[i]].feats.add(feats[i]);
        }
    }

private:
    std::vector<TrackData> &data;

    torch::Tensor nn_cosine_distance(torch::Tensor x, torch::Tensor y) {
        return std::get<0>(torch::min(1 - torch::matmul(x, y.t()), 0)).cpu();
    }
};

#endif //NN_MATCHING_H
