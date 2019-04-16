#include "DeepSORT.h"
#include "Extractor.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

DeepSORT::DeepSORT(const array<int64_t, 2> &dim)
        : extractor(make_unique<Extractor>()),
          manager(make_unique<TrackerManager>(dim, 0.2f)),
          feat_metric(make_unique<FeatureMetric>()) {}


DeepSORT::~DeepSORT() = default;

vector<Track> DeepSORT::update(const std::vector<cv::Rect2f> &dets, cv::Mat ori_img) {
    manager->predict();
    auto ret = manager->update(
            dets,
            [this, &dets, &ori_img]() {
                vector<cv::Rect2f> trks;
                vector<int> ids;
                for (auto &t : manager->trackers) {
                    trks.push_back(t.get_state());
                    ids.push_back(t.id);
                }
                auto iou_mat = iou_dist(dets, trks);

                vector<cv::Mat> boxes;
                for (auto &d:dets) {
                    boxes.push_back(ori_img(d));
                }
                auto feat_mat = feat_metric->distance(extractor->extract(boxes), ids);
                feat_mat.masked_fill_(iou_mat > 0.8f, 1E5f);
                return feat_mat;
            });

    vector<cv::Mat> boxes;
    vector<int> active_targets;
    for (auto[x, y]:manager->matched) {
        active_targets.push_back(x);
        boxes.push_back(ori_img(dets[y]));
    }
    vector<int> remain_targets;
    for (auto &t : manager->trackers) {
        remain_targets.push_back(t.id);
    }
    auto features = extractor->extract(boxes);
    feat_metric->update(features, active_targets, remain_targets);

    return ret;
}
