#include <set>
#include <algorithm>

#include "TrackerManager.h"
#include "Hungarian.h"

using namespace std;
using namespace cv;

void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                          vector<int> &unmatched_trks,
                                          vector<int> &unmatched_dets,
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
        if (dist_v[i][assignment[i]] > INVALID_DIST / 10) {
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
