#ifndef TARGET_H
#define TARGET_H

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <GL/gl3w.h>

#include "util.h"

using Frame = std::pair<int, cv::Rect2f>;

struct Target {
    Target() : snapshot_tex(0) {};

    explicit Target(const Frame &f, const cv::Mat &ss = cv::Mat())
            : snapshot(ss) {
        trajectories.insert(f);
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

    std::map<int, cv::Rect2f> trajectories;
    cv::Mat snapshot;
    GLuint snapshot_tex;
};

class TargetRepo {
public:
    using size_type = std::vector<Target>::size_type;

    size_type size() { return targets.size(); }

    const Target &operator[](size_type idx) { return targets[idx].first; }

    void update(const std::vector<Track> &trks,
                int frame, const cv::Mat &image);

    void merge(size_type to, size_type from);

    void erase(size_type idx);

private:
    static constexpr float padding = 0.1f;

    using TargetWrap = std::pair<Target, std::vector<int>>;

    std::vector<TargetWrap> targets;
    std::vector<int> discard_trks;
};

#endif //TARGET_H
