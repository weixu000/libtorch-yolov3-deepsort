#ifndef TARGET_H
#define TARGET_H

#include <vector>
#include <map>
#include <utility>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <GL/gl3w.h>

inline void mat_to_texture(const cv::Mat &mat, GLuint texture) {
    assert(mat.isContinuous());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB,
                 mat.cols, mat.rows,
                 0,
                 GL_BGR, GL_UNSIGNED_BYTE, mat.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

struct Snapshot {
    explicit Snapshot(const cv::Mat &mat = cv::Mat())
            : mat(mat), tex(0) {
        if (!mat.empty()) {
            glGenTextures(1, &tex);
            mat_to_texture(mat, tex);
        }
    }

    ~Snapshot() {
        glDeleteTextures(1, &tex);
    }

    Snapshot(const Snapshot &) = delete;

    Snapshot &operator=(const Snapshot &) = delete;

    Snapshot &operator=(Snapshot &&s) {
        if (this != &s) {
            mat = std::move(s.mat);
            tex = s.tex;

            s.tex = 0;
            s.mat = cv::Mat();
        }
    }

    Snapshot(Snapshot &&s) : Snapshot() {
        *this = std::move(s);
    }

    cv::Mat mat;
    GLuint tex;
};

struct Target {
    std::map<int, cv::Rect2f> trajectories;
    std::map<int, Snapshot> snapshots;
};

class TargetRepo {
public:
    using size_type = std::vector<Target>::size_type;

    size_type size() { return targets.size(); }

    const Target &operator[](size_type idx) { return targets[idx].first; }

    int load();

private:
    static constexpr float padding = 0.1f;

    using TargetWrap = std::pair<Target, std::vector<int>>;

    std::vector<TargetWrap> targets;
    std::vector<int> discard_trks;
    std::map<int, std::ifstream> trks_files;
};

#endif //TARGET_H
