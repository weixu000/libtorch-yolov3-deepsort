#include <experimental/filesystem>
#include <vector>
#include <algorithm>
#include <iterator>

#include "TargetRepo.h"

using namespace std;
namespace fs = std::experimental::filesystem;

int TargetRepo::load() {
    map<int, int> trk_tgt_map;
    for (int i = 0; i < targets.size(); ++i) {
        for (auto j:targets[i].second) {
            assert(trk_tgt_map.count(j) == 0);
            trk_tgt_map[j] = i;
        }
    }
    for (auto i:discard_trks) {
        assert(trk_tgt_map.count(i) == 0);
        trk_tgt_map[i] = -1;
    }

    vector<fs::path> result_dir;
    transform(fs::directory_iterator("result"), fs::directory_iterator(),
              back_inserter(result_dir),
              [](const fs::directory_entry &e) {
                  return e.path();
              });
    result_dir.erase(remove_if(result_dir.begin(), result_dir.end(),
                               [](const fs::path &p) { return !fs::is_directory(p); }),
                     result_dir.end());
    sort(result_dir.begin(), result_dir.end(), [](const fs::path &p1, const fs::path &p2) {
        return stoi(p1.filename()) < stoi(p2.filename());
    });

    for (auto &trk_dir: result_dir) {
        auto id = stoi(trk_dir.filename());
        if (!trk_tgt_map.count(id)) { // new track is target
            trks_files.emplace(id, ifstream(trk_dir / "trajectories.txt"));

            TargetWrap wrap;
            wrap.second = {id};
            targets.push_back(move(wrap));
            trk_tgt_map[id] = targets.size() - 1;
        }
        if (trk_tgt_map[id] != -1) { // add track to target
            auto &t = targets[trk_tgt_map[id]].first;

            for (auto &ss_p: fs::directory_iterator(trk_dir)) {
                auto &ss_path = ss_p.path();
                if (ss_path.extension() != ".txt" && !t.snapshots.count(stoi(ss_path.stem()))) {
                    auto img = cv::Mat();
                    try {
                        img = cv::imread(ss_path.string());
                    } catch (...) {
                        continue;
                    }
                    t.snapshots[stoi(ss_path.stem())] = Snapshot(img);
                }
            }

            int frame;
            cv::Rect2f box;
            while (trks_files[id] >> frame >> box.x >> box.y >> box.width >> box.height) {
                t.trajectories[frame] = box;
            }
            trks_files[id].clear(); // clear EOF flag
        }
    }

    int processed = 0;
    for (auto &t:targets) {
        processed = max(processed, t.first.trajectories.rbegin()->first);
    }
    return processed;
}
