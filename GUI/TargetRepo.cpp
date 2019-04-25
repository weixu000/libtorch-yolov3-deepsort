#include <experimental/filesystem>
#include <vector>
#include <algorithm>
#include <iterator>
#include <GL/gl3w.h>

#include "TargetRepo.h"

using namespace std;
namespace fs = std::experimental::filesystem;

void TargetRepo::load() {
    while (!stop) {
        for (auto &trk_dir: fs::directory_iterator(fs::path(output_dir) / "targets")) {
            auto id = stoi(trk_dir.path().filename());
            if (!targets.count(id)) { // new track is target
                trks_files.emplace(id, ifstream(trk_dir / "trajectories.txt"));
                {
                    lock_guard<mutex> lock(targets_mutex);
                    targets.emplace(id, Target());
                }
            }
            auto &t = targets[id];

            for (auto &ss_p: fs::directory_iterator(trk_dir / "snapshots")) {
                auto &ss_path = ss_p.path();
                if (!t.snapshots_tex.count(stoi(ss_path.stem()))) {
                    auto img = cv::Mat();
                    try {
                        img = cv::imread(ss_path.string());
                    } catch (...) {
                        continue;
                    }
                    {
                        lock_guard<mutex> lock(targets_mutex);
                        t.snapshots[stoi(ss_path.stem())] = img;
                    }
                }
            }

            int frame;
            cv::Rect2f box;
            while (trks_files[id] >> frame >> box.x >> box.y >> box.width >> box.height) {
                {
                    lock_guard<mutex> lock(targets_mutex);
                    t.trajectories[frame] = box;
                }
            }
            trks_files[id].clear(); // clear EOF flag
        }
    }
}

const TargetRepo::container_t &TargetRepo::get() const {
    targets_mutex.lock();

    // upload OpenGL textures in GUI thread
    // TODO: refactor it?
    for (auto &[id, t]:targets) {
        for (auto[frame, img]:t.snapshots) {
            if (t.snapshots_tex[frame].texId() == 0) {
                t.snapshots_tex[frame].copyFrom(img, true);
            }
        }
    }
    return targets;
}

void TargetRepo::finished_get() const {
    // need to call after use of get()
    // TODO: better approach?
    targets_mutex.unlock();
}

int TargetRepo::processed() const {
    int processed_frame = 0;
    for (auto &[id, t]:get()) {
        if (t.trajectories.empty()) continue;
        processed_frame = max(processed_frame, t.trajectories.rbegin()->first);
    }
    finished_get();
    return processed_frame;
}
