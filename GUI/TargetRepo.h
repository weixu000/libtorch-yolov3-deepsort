#ifndef TARGET_H
#define TARGET_H

#include <map>
#include <fstream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>

struct Target {
    std::map<int, cv::Rect2f> trajectories;
    std::map<int, cv::ogl::Texture2D> snapshots_tex;
    std::map<int, cv::Mat> snapshots;
};

class TargetRepo {
public:
    TargetRepo() : load_thread(&TargetRepo::load, this) {}

    ~TargetRepo() {
        stop = true;
        load_thread.join();
    }

    using container_t = std::map<int, Target>;

    const container_t &get() const;

    void finished_get() const;

    int processed() const;

private:
    static inline const std::string output_dir = "result";

    std::thread load_thread;

    void load();

    volatile bool stop = false;

    mutable container_t targets;
    mutable std::mutex targets_mutex;

    std::map<int, std::ifstream> trks_files;
};

#endif //TARGET_H
