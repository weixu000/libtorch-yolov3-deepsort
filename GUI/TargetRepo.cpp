#include <experimental/filesystem>
#include <vector>

#include "TargetRepo.h"
#include "config.h"

using namespace std;
namespace fs = std::experimental::filesystem;

void TargetRepo::load(const std::function<void(int)> &show_progress) {
    int num_targets = 0;
    for (auto &trk_dir: fs::directory_iterator(fs::path(out_dir) / TARGETS_DIR_NAME)) {
        ++num_targets;
    }

    int i_target = 0;
    for (auto &trk_dir: fs::directory_iterator(fs::path(out_dir) / TARGETS_DIR_NAME)) {
        auto id = stoi(trk_dir.path().filename());
        targets.emplace(id, Target());
        auto &t = targets[id];

        for (auto &ss_p: fs::directory_iterator(trk_dir / SNAPSHOTS_DIR_NAME)) {
            auto &ss_path = ss_p.path();
            if (!t.snapshots.count(stoi(ss_path.stem()))) {
                auto img = cv::imread(ss_path.string());
                t.snapshots[stoi(ss_path.stem())] = img;
                assert(!img.empty());
            }
        }

        auto traj_txt = ifstream((trk_dir / TRAJ_TXT_NAME).string());
        int frame;
        cv::Rect2f box;
        while (traj_txt >> frame >> box.x >> box.y >> box.width >> box.height) {
            t.trajectories[frame] = box;
        }
        traj_txt.close();

        show_progress(100 * ++i_target / num_targets);
    }
}

std::string TargetRepo::video_path() {
    return (fs::path(out_dir) / VIDEO_NAME).string();
}
