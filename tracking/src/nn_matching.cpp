#include <algorithm>

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
