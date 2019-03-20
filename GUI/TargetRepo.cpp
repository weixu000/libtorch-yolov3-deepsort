#include <experimental/filesystem>

#include "TargetRepo.h"

using namespace std;
namespace fs=std::experimental::filesystem;

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

    for (auto &trk_dir: fs::directory_iterator("targets")) {
        auto id = stoi(trk_dir.path().filename());
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
                    t.snapshots[stoi(ss_path.stem())] = Snapshot(cv::imread(ss_path.string()));
                }
            }

            int frame;
            cv::Rect2f box;
            while (trks_files[id] >> frame >> box.x >> box.y >> box.width >> box.height) {
                t.trajectories[frame] = box;
            }
        }
    }

    int processed = 0;
    for (auto &t:targets) {
        processed = max(processed, t.first.trajectories.rbegin()->first);
    }
    return processed;
}

void TargetRepo::merge(TargetRepo::size_type to, TargetRepo::size_type from) {
    auto &[target_to, trks_to] = targets[to];
    auto &[target_from, trks_from] = targets[from];

    // merge trajectories
    // the target merge from has higher priority
    target_from.trajectories.merge(target_to.trajectories);
    target_to.trajectories = move(target_from.trajectories);

    // merge snapshots
    target_from.snapshots.merge(target_to.snapshots);
    target_to.snapshots = move(target_from.snapshots);

    // merged target should have all tracks
    trks_to.insert(trks_to.end(), trks_from.begin(), trks_from.end());

    targets.erase(targets.begin() + from);
}

void TargetRepo::erase(TargetRepo::size_type idx) {
    for (auto i:targets[idx].second) {
        discard_trks.push_back(i);
        trks_files.erase(i);
    }
    targets.erase(targets.begin() + idx); // delete the target
}
