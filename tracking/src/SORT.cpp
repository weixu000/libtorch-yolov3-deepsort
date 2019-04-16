#include "SORT.h"
#include "TrackerManager.h"
#include "nn_matching.h"

using namespace std;

SORT::SORT(const array<int64_t, 2> &dim)
        : manager(make_unique<TrackerManager>(dim, 0.7f)) {}

SORT::~SORT() = default;

vector<Track> SORT::update(const vector<cv::Rect2f> &dets) {
    manager->predict();
    auto ret = manager->update(
            dets,
            [this, &dets]() {
                vector<cv::Rect2f> trks;
                for (auto &t : manager->trackers) {
                    trks.push_back(t.get_state());
                }
                return iou_dist(dets, trks);
            });

    return ret;
}
