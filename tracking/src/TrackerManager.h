#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <torch/torch.h>
#include <tuple>

#include "Track.h"
#include "KalmanTracker.h"

using DistanceMetricFunc = std::function<
        torch::Tensor(const std::vector<int> &trk_ids, const std::vector<int> &det_ids)>;

const float INVALID_DIST = 1E3f;

void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                          std::vector<int> &unmatched_trks,
                                          std::vector<int> &unmatched_dets,
                                          std::vector<std::tuple<int, int>> &matched);

template<typename TrackData>
class TrackerManager {
public:
    explicit TrackerManager(std::vector<TrackData> &data, const std::array<int64_t, 2> &dim)
            : data(data), img_box(0, 0, dim[1], dim[0]) {}

    void predict() {
        for (auto &t:data) {
            t.kalman.predict();
        }
    }

    void remove_nan() {
        data.erase(remove_if(data.begin(), data.end(),
                             [](const TrackData &t) {
                                 auto bbox = t.kalman.rect();
                                 return std::isnan(bbox.x) || std::isnan(bbox.y) ||
                                        std::isnan(bbox.width) || std::isnan(bbox.height);
                             }),
                   data.end());
    }

    void remove_deleted() {
        data.erase(remove_if(data.begin(), data.end(),
                             [this](const TrackData &t) {
                                 return t.kalman.state() == TrackState::Deleted;
                             }), data.end());
    }

    std::vector<std::tuple<int, int>>
    update(const std::vector<cv::Rect2f> &dets,
           const DistanceMetricFunc &confirmed_metric, const DistanceMetricFunc &unconfirmed_metric) {
        std::vector<int> unmatched_trks;
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Confirmed) {
                unmatched_trks.emplace_back(i);
            }
        }

        std::vector<int> unmatched_dets(dets.size());
        iota(unmatched_dets.begin(), unmatched_dets.end(), 0);

        std::vector<std::tuple<int, int>> matched;

        associate_detections_to_trackers_idx(confirmed_metric, unmatched_trks, unmatched_dets, matched);

        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Tentative) {
                unmatched_trks.emplace_back(i);
            }
        }

        associate_detections_to_trackers_idx(unconfirmed_metric, unmatched_trks, unmatched_dets, matched);

        for (auto i : unmatched_trks) {
            data[i].kalman.miss();
        }

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a manager
        for (auto[x, y] : matched) {
            data[x].kalman.update(dets[y]);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatched_dets) {
            matched.emplace_back(data.size(), umd);
            auto t = TrackData{};
            t.kalman.init(dets[umd]);
            data.emplace_back(t);
        }

        return matched;
    }

    std::vector<Track> visible_tracks() {
        std::vector<Track> ret;
        for (auto &t : data) {
            auto bbox = t.kalman.rect();
            if (t.kalman.state() == TrackState::Confirmed &&
                img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
                Track res{t.kalman.id(), bbox};
                ret.push_back(res);
            }
        }
        return ret;
    }

private:
    std::vector<TrackData> &data;
    const cv::Rect2f img_box;
};

#endif //TRACKER_H
