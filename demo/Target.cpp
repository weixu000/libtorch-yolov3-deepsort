#include "Target.h"


void TargetRepo::update(const std::vector<Track> &trks, int frame, const cv::Mat &image) {
    std::map<int, int> trk_tgt_map;
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

    for (auto &[id, box]:trks) {
        if (!trk_tgt_map.count(id)) { // new track is target
            TargetWrap wrap;
            wrap.first.trajectories[frame] = box;
            wrap.first.snapshots[frame] = std::move(Snapshot(image(unnormalize_rect(pad_rect(box, padding), image.cols,
                                                                                    image.rows)).clone()));
            wrap.second = {id};
            targets.push_back(std::move(wrap));
        } else if (trk_tgt_map[id] != -1) { // add track to target
            auto &t = targets[trk_tgt_map[id]].first;
            t.trajectories.emplace(frame, box);
            if (frame - t.snapshots.rbegin()->first > 5) {
                t.snapshots[frame] = std::move(Snapshot(image(unnormalize_rect(pad_rect(box, padding), image.cols,
                                                                               image.rows)).clone()));
            }
        }
    }
}

void TargetRepo::merge(TargetRepo::size_type to, TargetRepo::size_type from) {
    auto &[target_to, trks_to] = targets[to];
    auto &[target_from, trks_from] = targets[from];

    // merge trajectories
    // the target merge from has higher priority
    target_from.trajectories.merge(target_to.trajectories);
    target_to.trajectories = std::move(target_from.trajectories);

    // merge snapshots
    target_from.snapshots.merge(target_to.snapshots);
    target_to.snapshots = std::move(target_from.snapshots);

    // merged target should have all tracks
    trks_to.insert(trks_to.end(), trks_from.begin(), trks_from.end());

    targets.erase(targets.begin() + from);
}

void TargetRepo::erase(TargetRepo::size_type idx) {
    for (auto i:targets[idx].second) {
        discard_trks.push_back(i);
    }
    targets.erase(targets.begin() + idx); // delete the target
}
