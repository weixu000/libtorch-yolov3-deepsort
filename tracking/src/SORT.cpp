#include "SORT.h"
#include "Tracker.h"
#include "nn_matching.h"

using namespace std;

SORT::SORT(const array<int64_t, 2> &dim)
        : tracker(make_unique<Tracker>(dim)) {}

SORT::~SORT() = default;

vector<Track> SORT::update(const vector<cv::Rect2f> &dets) {
    auto ret = tracker->update(
            dets,
            [this, &dets]() {
                vector<cv::Rect2f> trks;
                for (auto &t : tracker->trackers) {
                    trks.push_back(t.get_state());
                }
                return iou_dist(dets, trks);
            });

    return ret;
}
