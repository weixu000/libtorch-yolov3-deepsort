#ifndef TARGET_H
#define TARGET_H

#include <vector>
#include <map>
#include <utility>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>

struct Target {
    std::map<int, cv::Rect2f> trajectories;
    std::map<int, cv::ogl::Texture2D> snapshots;
};

class TargetRepo {
public:
    using size_type = std::vector<Target>::size_type;

    size_type size() { return targets.size(); }

    const Target &operator[](size_type idx) { return targets[idx]; }

    int load();

private:
    static inline const float padding = 0.1f;

    static inline const std::string output_dir = "result";

    std::map<size_type, Target> targets;
    std::map<size_type, std::ifstream> trks_files;
};

#endif //TARGET_H
