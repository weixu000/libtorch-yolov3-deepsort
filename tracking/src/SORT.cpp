#include "SORT.h"
#include "Tracker.h"
#include "nn_matching.h"

using namespace std;

SORT::SORT(const array<int64_t, 2> &dim)
        : tracker(make_unique<Tracker>(dim)) {}

SORT::~SORT() = default;

vector<Track> SORT::update(const vector<cv::Rect2f> &dets) {
    auto ret = tracker->update(dets, iou_dist);

    return ret;
}
