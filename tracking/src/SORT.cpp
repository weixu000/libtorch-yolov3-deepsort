#include "SORT.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

SORT::SORT(const array<int64_t, 2> &dim)
        : manager(make_unique<TrackerManager>(dim)) {}

SORT::~SORT() = default;

vector<Track> SORT::update(const vector<cv::Rect2f> &detections) {
    manager->predict();
    auto metric = [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
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
    };
    manager->update(detections, metric, metric);

    return manager->visible_tracks();
}
