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
    using size_type=std::vector<Target>::size_type;

    size_type size() { return targets.size(); }

    const Target &operator[](size_type idx) {
        return targets[idx];
    }

    void update(const std::vector<Track> &trks,
                int frame, const cv::Mat &image) {
        for (auto &[id, box]:trks) {
            if (!trk_tgt_map.count(id)) { // new track is target
                trk_tgt_map.emplace(id, targets.size());
                targets.emplace_back(std::make_pair(frame, box),
                                     image(unnormalize_rect(box, image.cols, image.rows)).clone());
            } else if (trk_tgt_map[id] != -1) { // add track to target
                targets[trk_tgt_map[id]].trajectories.emplace_back(frame, box);
            }
        }
    }

    void merge(size_type to, size_type from) {
        find_if(trk_tgt_map.begin(), trk_tgt_map.end(),
                [from](const std::pair<int, int> &x) {
                    return x.second == from;
                })->second = to; // re-map the track
        auto mid = targets[to].trajectories.insert(targets[to].trajectories.end(),
                                                   targets[from].trajectories.begin(),
                                                   targets[from].trajectories.end());
        inplace_merge(targets[to].trajectories.begin(), mid, targets[to].trajectories.end(),
                      [](const Frame &a, const Frame &b) {
                          return a.first < b.first;
                      });
        // TODO: handle different boxes in the same fame
    }

    void erase(size_type idx) {
        targets.erase(targets.begin() + idx); // delete the target
        // TODO: targets' idx will change after erase()
        find_if(trk_tgt_map.begin(), trk_tgt_map.end(),
                [idx](const std::pair<int, int> &x) {
                    return x.second == idx;
                })->second = -1; // discard the track
    }

private:
    std::vector<Target> targets;
    std::map<int, int> trk_tgt_map;
};

#endif //TARGET_H
