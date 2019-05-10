#include <experimental/filesystem>
#include <vector>

#include "TargetRepo.h"
#include "config.h"

using namespace std;
namespace fs = std::experimental::filesystem;

void TargetRepo::load(const std::function<void(int)> &show_progress) {
    int num_targets = 0;
    for (auto &trk_dir: fs::directory_iterator(fs::path(OUTPUT_DIR) / TARGETS_DIR_NAME)) {
        ++num_targets;
    }

    int i_target = 0;
    for (auto &trk_dir: fs::directory_iterator(fs::path(OUTPUT_DIR) / TARGETS_DIR_NAME)) {
        auto id = stoi(trk_dir.path().filename());
        if (!targets.count(id)) { // new track is target
            trks_files.emplace(id, ifstream(trk_dir / TRAJ_TXT_NAME));
            targets.emplace(id, Target());
        }
        auto &t = targets[id];

        for (auto &ss_p: fs::directory_iterator(trk_dir / SNAPSHOTS_DIR_NAME)) {
            auto &ss_path = ss_p.path();
            if (!t.snapshots.count(stoi(ss_path.stem()))) {
                auto img = cv::Mat();
                try {
                    img = cv::imread(ss_path.string());
                } catch (...) {
                    continue;
                }
                t.snapshots[stoi(ss_path.stem())] = img;
            }
        }

        int frame;
        cv::Rect2f box;
        while (trks_files[id] >> frame >> box.x >> box.y >> box.width >> box.height) {
            t.trajectories[frame] = box;
        }
        trks_files[id].clear(); // clear EOF flag

        show_progress(100 * ++i_target / num_targets);
    }
}

int TargetRepo::processed() const {
    int processed_frame = 0;
    for (auto &[id, t]:targets) {
        if (t.trajectories.empty()) continue;
        processed_frame = max(processed_frame, t.trajectories.rbegin()->first);
    }
    return processed_frame;
}
