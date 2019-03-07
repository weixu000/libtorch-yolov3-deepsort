#ifndef TARGET_H
#define TARGET_H

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

#include "util.h"

using Frame = std::pair<int, cv::Rect2f>;

struct Target {
    Target() : snapshot_tex(0) {};

    explicit Target(const Frame &f, const cv::Mat &ss = cv::Mat())
            : snapshot(ss) {
        trajectories.push_back(f);
        if (!snapshot.empty()) {
            glGenTextures(1, &snapshot_tex);
            mat_to_texture(snapshot, snapshot_tex);
        }
    }

    ~Target() {
        glDeleteTextures(1, &snapshot_tex);
    }

    Target(const Target &) = delete;

    Target &operator=(const Target &) = delete;

    Target(Target &&t)
            : trajectories(std::move(t.trajectories)),
              snapshot(std::move(t.snapshot)) {
        snapshot_tex = t.snapshot_tex;
        t.snapshot_tex = 0;
    }

    Target &operator=(Target &&t) {
        trajectories = std::move(t.trajectories);
        snapshot = std::move(t.snapshot);
        snapshot_tex = t.snapshot_tex;
        t.snapshot_tex = 0;
    }

    std::vector<Frame> trajectories;
    cv::Mat snapshot;
    GLuint snapshot_tex;
};

class TargetRepo {
public:
    using size_type = std::vector<Target>::size_type;

    size_type size() { return targets.size(); }

    const Target &operator[](size_type idx) {
        return targets[idx].first;
    }

    void update(const std::vector<Track> &trks,
                int frame, const cv::Mat &image) {
        std::map<int, int> trk_tgt_map;
        for (int i = 0; i < targets.size(); ++i) {
            for (auto j:targets[i].second) {
                assert(trk_tgt_map.count(j) == 0);
                trk_tgt_map[j] = i;
            }
        }
        for (auto i:discard_trks) {
            assert(trk_tgt_map.count(i) == 0);
            trk_tgt_map[i] = -1;
        }

        for (auto &[id, box]:trks) {
            if (!trk_tgt_map.count(id)) { // new track is target
                TargetWrap wrap;
                wrap.first = Target(std::make_pair(frame, box),
                                    image(unnormalize_rect(box, image.cols, image.rows)).clone());
                wrap.second = {id};
                targets.push_back(std::move(wrap));
            } else if (trk_tgt_map[id] != -1) { // add track to target
                targets[trk_tgt_map[id]].first.trajectories.emplace_back(frame, box);
            }
        }
    }

    void merge(size_type to, size_type from) {
        auto &[target_to, trks_to] = targets[to];
        auto &[target_from, trks_from] = targets[from];
        auto mid = target_to.trajectories.insert(target_to.trajectories.end(),
                                                 target_from.trajectories.begin(),
                                                 target_from.trajectories.end());
        inplace_merge(target_to.trajectories.begin(), mid, target_to.trajectories.end(),
                      [](const Frame &a, const Frame &b) {
                          return a.first < b.first;
                      });

        trks_to.insert(trks_to.end(), trks_from.begin(), trks_from.end());
        targets.erase(targets.begin() + from);
        // TODO: handle different boxes in the same fame
    }

    void erase(size_type idx) {
        for (auto i:targets[idx].second) {
            discard_trks.push_back(i);
        }
        targets.erase(targets.begin() + idx); // delete the target
    }

private:
    using TargetWrap = std::pair<Target, std::vector<int>>;

    std::vector<TargetWrap> targets;
    std::vector<int> discard_trks;
};

#endif //TARGET_H
