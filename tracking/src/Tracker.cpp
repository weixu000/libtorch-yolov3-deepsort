#include <set>
#include <algorithm>
#include <tuple>

#include "Tracker.h"
#include "Hungarian.h"

using namespace std;
using namespace cv;

// Computes IOU between two bounding boxes
static inline float iou(const Rect2f &bb_test, const Rect2f &bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}

static inline auto associate_detections_to_trackers(const vector<Rect2f> &dets,
                                                    const vector<Rect2f> &trks,
                                                    float threshold) {
    auto trk_num = trks.size();
    auto det_num = dets.size();
    vector<vector<double>> iou_mat(trk_num, vector<double>(det_num, 0));
    for (unsigned int i = 0; i < trk_num; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < det_num; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iou_mat[i][j] = 1 - iou(trks[i], dets[j]);
        }
    }

    vector<int> assignment;
    HungarianAlgorithm().Solve(iou_mat, assignment);

    set<int> unmatched_dets;
    set<int> unmatched_trks;
    if (trk_num < det_num) {
        set<int> allItems, matchedItems;
        for (unsigned int n = 0; n < det_num; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trk_num; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       inserter(unmatched_dets, unmatched_dets.begin()));
    } else if (det_num < trk_num) {
        for (unsigned int i = 0; i < trk_num; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatched_trks.insert(i);
    }

    // filter out matched with low IOU
    vector<cv::Point> matched;
    for (unsigned int i = 0; i < trk_num; ++i) {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iou_mat[i][assignment[i]] < threshold) {
            unmatched_trks.insert(i);
            unmatched_dets.insert(assignment[i]);
        } else
            matched.emplace_back(i, assignment[i]);
    }

    return make_tuple(matched, unmatched_dets, unmatched_trks);
}

const int Tracker::max_age = 10;
const int Tracker::min_hits = 3;
const float Tracker::iou_threshold = 0.3;

vector<Track> Tracker::update(const vector<Rect2f> &dets) {
    ++frame_count;

    for (auto &t:trackers) {
        t.predict();
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [](const KalmanTracker &t) {
                                 auto bbox = t.get_state();
                                 return !(bbox.x >= 0 && bbox.y >= 0);
                             }),
                   trackers.end());

    vector<Rect2f> trks;
    for (auto &t : trackers) {
        trks.push_back(t.get_state());
    }

    auto[matched, unmatched_dets, unmatched_trks] = associate_detections_to_trackers(dets,
                                                                                     trks,
                                                                                     iou_threshold);

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    for (auto &m : matched) {
        trackers[m.x].update(dets[m.y]);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatched_dets) {
        trackers.emplace_back(dets[umd]);
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [this](const KalmanTracker &t) {
                                 return t.time_since_update > max_age;
                             }),
                   trackers.end());

    vector<Track> ret;
    for (auto &t : trackers) {
        if (t.hit_streak >= min_hits || frame_count <= min_hits) {
            Track res;
            res.box = t.get_state();
            res.id = t.id + 1;
            ret.push_back(res);
        }
    }
    return ret;
}
