#include <set>
#include <algorithm>
#include <tuple>

#include "Tracker.h"
#include "Hungarian.h"

using namespace std;
using namespace cv;

namespace {
    auto associate_detections_to_trackers(torch::Tensor dist, float threshold = 0.7f) {
        auto trk_num = dist.size(0);
        auto det_num = dist.size(1);
        auto dist_a = dist.accessor<float, 2>();
        auto dist_v = vector<vector<double>>(dist.size(0), vector<double>(dist.size(1)));
        for (size_t i = 0; i < dist.size(0); ++i) {
            for (size_t j = 0; j < dist.size(1); ++j) {
                dist_v[i][j] = dist_a[i][j];
            }
        }

        vector<int> assignment;
        HungarianAlgorithm().Solve(dist_v, assignment);

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
            if (dist_v[i][assignment[i]] > threshold) {
                unmatched_trks.insert(i);
                unmatched_dets.insert(assignment[i]);
            } else
                matched.emplace_back(i, assignment[i]);
        }

        return make_tuple(matched, unmatched_dets, unmatched_trks);
    }
}

vector<Track> Tracker::update(const vector<Rect2f> &dets,
                              const DistanceMetricFunc &metric) {
    ++frame_count;

    for (auto &t:trackers) {
        t.predict();
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [](const KalmanTracker &t) {
                                 auto bbox = t.get_state();
                                 return std::isnan(bbox.x) || std::isnan(bbox.y)
                                        || std::isnan(bbox.width) || std::isnan(bbox.height);
                             }),
                   trackers.end());

    vector<Rect2f> trks;
    for (auto &t : trackers) {
        trks.push_back(t.get_state());
    }

    auto dist = metric(dets, trks);
    auto[matched, unmatched_dets, unmatched_trks] = associate_detections_to_trackers(dist);

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
        auto bbox = t.get_state();
        if (img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
            Track res{t.id, bbox};
            ret.push_back(res);
        }
    }
    return ret;
}
