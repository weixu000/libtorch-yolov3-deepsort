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

#endif //TARGET_H
