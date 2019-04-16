#include "nn_matching.h"

using namespace std;
using namespace cv;

namespace {
    float iou(const Rect2f &bb_test, const Rect2f &bb_gt) {
        auto in = (bb_test & bb_gt).area();
        auto un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return in / un;
    }

    torch::Tensor nn_cosine_distance(torch::Tensor x, torch::Tensor y) {
        return get<0>(torch::min(1 - torch::matmul(x, y.t()), 0));
    }
}

torch::Tensor FeatureMetric::distance(torch::Tensor features, const vector<int> &targets) {
    auto dist = torch::empty({int64_t(targets.size()), features.size(0)});
    if (features.size(0)) {
        for (size_t i = 0; i < targets.size(); ++i) {
            dist[i] = nn_cosine_distance(samples.at(targets[i]).get(), features);
        }
    }

    return dist;
}

void FeatureMetric::update(torch::Tensor feats, const vector<int> &active_targets,
                           const vector<int> &remain_targets) {
    map<int, FeatureBundle> samples_r;
    for (auto t:remain_targets) {
        if (samples.count(t)) {
            samples_r.emplace(t, samples[t]);
        }
    }
    samples = move(samples_r);

    for (size_t i = 0; i < active_targets.size(); ++i) {
        samples[active_targets[i]].add(feats[i]);
    }
}

void FeatureMetric::FeatureBundle::add(torch::Tensor feat) {
    if (next == budget) {
        full = true;
        next = 0;
    }
    store[next++] = feat;
}

torch::Tensor FeatureMetric::FeatureBundle::get() {
    return full ? store : store.slice(0, 0, next);
}

torch::Tensor iou_dist(const vector<Rect2f> &dets, const vector<Rect2f> &trks) {
    auto trk_num = trks.size();
    auto det_num = dets.size();
    auto dist = torch::empty({int64_t(trk_num), int64_t(det_num)});
    for (size_t i = 0; i < trk_num; i++) // compute iou matrix as a distance matrix
    {
        for (size_t j = 0; j < det_num; j++) {
            dist[i][j] = 1 - iou(trks[i], dets[j]);
        }
    }
    return dist;
}
