#include <algorithm>

#include "DeepSORT.h"
#include "Extractor.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

DeepSORT::DeepSORT(const array<int64_t, 2> &dim)
        : extractor(make_unique<Extractor>()),
          manager(make_unique<TrackerManager>(dim)),
          feat_metric(make_unique<FeatureMetric>()) {}


DeepSORT::~DeepSORT() = default;

vector<Track> DeepSORT::update(const std::vector<cv::Rect2f> &detections, cv::Mat ori_img) {
    feat_metric->erase(manager->predict());

    auto[matched, removed]= manager->update(
            detections,
            [this, &detections, &ori_img](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(manager->trackers[t].rect());
                }
                vector<cv::Mat> boxes;
                vector<cv::Rect2f> dets;
                for (auto d:det_ids) {
                    dets.push_back(detections[d]);
                    boxes.push_back(ori_img(detections[d]));
                }

                auto iou_mat = iou_dist(dets, trks);
                auto feat_mat = feat_metric->distance(extractor->extract(boxes), trk_ids);
                feat_mat.masked_fill_(iou_mat > 0.8f, TrackerManager::invalid_dist);
                feat_mat.masked_fill_(feat_mat > 0.2f, TrackerManager::invalid_dist);
                return feat_mat;
            },
            [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(manager->trackers[t].rect());
                }
                vector<cv::Rect2f> dets;
                for (auto &d:det_ids) {
                    dets.push_back(detections[d]);
                }
                auto iou_mat = iou_dist(dets, trks);
                iou_mat.masked_fill_(iou_mat > 0.7f, TrackerManager::invalid_dist);
                return iou_mat;
            });

    vector<cv::Mat> boxes;
    vector<int> targets;
    for (auto[x, y]:matched) {
        targets.emplace_back(x);
        boxes.emplace_back(ori_img(detections[y]));
    }
    feat_metric->update(extractor->extract(boxes), targets);
    feat_metric->erase(removed);

    assert(feat_metric->samples.size() == manager->trackers.size());

    return manager->visible_tracks();
}
