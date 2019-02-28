#include <set>
#include <algorithm>

#include "Tracker.h"
#include "Hungarian.h"

using namespace std;

// Computes IOU between two bounding boxes
float GetIOU(Rect2f bb_test, Rect2f bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}

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

    vector<Rect2f> predictedBoxes;
    for (auto &t : trackers) {
        predictedBoxes.push_back(t.get_state());
    }

    auto trkNum = trackers.size();
    auto detNum = dets.size();
    vector<vector<double>> iouMatrix(trkNum, vector<double>(detNum, 0));
    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], dets[j]);
        }
    }

    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    HungAlgo.Solve(iouMatrix, assignment);

    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    if (trkNum < detNum) //	there are unmatched detections
    {
        set<int> allItems, matchedItems;
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       inserter(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }

    // filter out matched with low IOU
    vector<cv::Point> matchedPairs;
    for (unsigned int i = 0; i < trkNum; ++i) {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else
            matchedPairs.emplace_back(i, assignment[i]);
    }

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    for (auto &m : matchedPairs) {
        auto trkIdx = m.x;
        auto detIdx = m.y;
        trackers[trkIdx].update(dets[detIdx]);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections) {
        auto tracker = KalmanTracker(dets[umd]);
        trackers.push_back(tracker);
    }

    trackers.erase(remove_if(trackers.begin(), trackers.end(),
                             [this](const KalmanTracker &t) {
                                 return t.m_time_since_update > max_age;
                             }),
                   trackers.end());

    vector<Track> trackingResult;
    for (auto &t : trackers) {
        if ((t.m_time_since_update < 1) &&
            (t.m_hit_streak >= min_hits || frame_count <= min_hits)) {
            Track res;
            res.box = t.get_state();
            res.id = t.m_id + 1;
            trackingResult.push_back(res);
        }
    }
    return trackingResult;
}
