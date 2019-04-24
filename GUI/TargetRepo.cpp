#include <experimental/filesystem>
#include <vector>
#include <algorithm>
#include <iterator>
#include <GL/gl3w.h>

#include "TargetRepo.h"

using namespace std;
namespace fs = std::experimental::filesystem;

int TargetRepo::load() {
    for (auto &trk_dir: fs::directory_iterator(fs::path(output_dir) / "targets")) {
        auto id = stoi(trk_dir.path().filename());
        if (!targets.count(id)) { // new track is target
            trks_files.emplace(id, ifstream(trk_dir / "trajectories.txt"));
            targets.emplace(id, Target());
        }
        auto &t = targets[id];

        for (auto &ss_p: fs::directory_iterator(trk_dir / "snapshots")) {
            auto &ss_path = ss_p.path();
            if (!t.snapshots.count(stoi(ss_path.stem()))) {
                auto img = cv::Mat();
                try {
                    img = cv::imread(ss_path.string());
                } catch (...) {
                    continue;
                }
                t.snapshots[stoi(ss_path.stem())] = cv::ogl::Texture2D(img, true);
            }
        }

        int frame;
        cv::Rect2f box;
        while (trks_files[id] >> frame >> box.x >> box.y >> box.width >> box.height) {
            t.trajectories[frame] = box;
        }
        trks_files[id].clear(); // clear EOF flag
    }

    int processed = 0;
    for (auto &t:targets) {
        if (t.second.trajectories.empty()) continue;
        processed = max(processed, t.second.trajectories.rbegin()->first);
    }
    return processed;
}
