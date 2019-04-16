#include <set>
#include <algorithm>

#include "TrackerManager.h"
#include "Hungarian.h"

using namespace std;
using namespace cv;

namespace {
    void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                              vector<int> &unmatched_trks,
                                              std::vector<int> &unmatched_dets,
                                              vector<tuple<int, int>> &matched) {
        auto dist = metric(unmatched_trks, unmatched_dets);
        auto dist_a = dist.accessor<float, 2>();
        auto dist_v = vector<vector<double>>(dist.size(0), vector<double>(dist.size(1)));
        for (size_t i = 0; i < dist.size(0); ++i) {
            for (size_t j = 0; j < dist.size(1); ++j) {
                dist_v[i][j] = dist_a[i][j];
            }
        }

        vector<int> assignment;
        HungarianAlgorithm().Solve(dist_v, assignment);

        // filter out matched with low IOU
        for (size_t i = 0; i < assignment.size(); ++i) {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (dist_v[i][assignment[i]] > TrackerManager::invalid_dist / 10) {
                assignment[i] = -1;
            } else {
                matched.emplace_back(make_tuple(unmatched_trks[i], unmatched_dets[assignment[i]]));
            }
        }

        for (size_t i = 0; i < assignment.size(); ++i) {
            if (assignment[i] != -1) {
                unmatched_trks[i] = -1;
            }
        }
        unmatched_trks.erase(remove_if(unmatched_trks.begin(), unmatched_trks.end(),
                                       [](int i) { return i == -1; }),
                             unmatched_trks.end());

        sort(assignment.begin(), assignment.end());
        vector<int> unmatched_dets_new;
        set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                       assignment.begin(), assignment.end(),
                       inserter(unmatched_dets_new, unmatched_dets_new.begin()));
        unmatched_dets = move(unmatched_dets_new);
    }
}

const float TrackerManager::invalid_dist = 1E3f;

vector<int> TrackerManager::predict() {
    for (auto &t:trackers) {
        t.predict();
    }

    vector<int> removed_trks;
    for (size_t i = 0; i < trackers.size(); ++i) {
        auto bbox = trackers[i].rect();
        if (std::isnan(bbox.x) || std::isnan(bbox.y) ||
            std::isnan(bbox.width) || std::isnan(bbox.height)) {
            removed_trks.emplace_back(i);
        }
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [](const KalmanTracker &t) {
                                 auto bbox = t.rect();
                                 return std::isnan(bbox.x) || std::isnan(bbox.y) ||
                                        std::isnan(bbox.width) || std::isnan(bbox.height);
                             }),
                   trackers.end());

    return removed_trks;
}

std::tuple<std::vector<std::tuple<int, int>>, std::vector<int>>
TrackerManager::update(const vector<Rect2f> &dets,
                       const DistanceMetricFunc &confirmed_metric, const DistanceMetricFunc &unconfirmed_metric) {
    vector<int> unmatched_trks;
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (trackers[i].state() == TrackState::Confirmed) {
            unmatched_trks.emplace_back(i);
        }
    }

    std::vector<int> unmatched_dets(dets.size());
    iota(unmatched_dets.begin(), unmatched_dets.end(), 0);

    vector<tuple<int, int>> matched;

    associate_detections_to_trackers_idx(confirmed_metric, unmatched_trks, unmatched_dets, matched);

    for (size_t i = 0; i < trackers.size(); ++i) {
        if (trackers[i].state() == TrackState::Tentative) {
            unmatched_trks.emplace_back(i);
        }
    }

    associate_detections_to_trackers_idx(unconfirmed_metric, unmatched_trks, unmatched_dets, matched);

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a manager
    for (auto[x, y] : matched) {
        trackers[x].update(dets[y]);
    }

    for (auto i : unmatched_trks) {
        trackers[i].miss();
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatched_dets) {
        matched.emplace_back(trackers.size(), umd);
        trackers.emplace_back(dets[umd]);
    }

    vector<int> removed_trks;
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (trackers[i].state() == TrackState::Deleted) {
            removed_trks.emplace_back(i);
        }
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [this](const KalmanTracker &t) {
                                 return t.state() == TrackState::Deleted;
                             }), trackers.end());

    return make_tuple(matched, removed_trks);
}

vector<Track> TrackerManager::visible_tracks() {
    vector<Track> ret;
    for (auto &t : trackers) {
        auto bbox = t.rect();
        if (t.state() == TrackState::Confirmed &&
            img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
            Track res{t.id(), bbox};
            ret.push_back(res);
        }
    }
    return ret;
}
