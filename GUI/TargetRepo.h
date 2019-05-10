#ifndef TARGET_H
#define TARGET_H

#include <map>
#include <functional>
#include <fstream>
#include <opencv2/opencv.hpp>

struct Target {
    std::map<int, cv::Rect2f> trajectories;
    std::map<int, cv::Mat> snapshots;
};

class TargetRepo {
public:
    using container_t = std::map<int, Target>;

    container_t &get() { return targets; }

    int processed() const;

    void load(const std::function<void(int)> &show_progress);

private:
    container_t targets;

    std::map<int, std::ifstream> trks_files;
};

#endif //TARGET_H
