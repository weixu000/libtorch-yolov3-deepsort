#include <experimental/filesystem>
#include <fstream>

#include "TargetRepo.h"

using namespace std;
namespace fs=std::experimental::filesystem;

void TargetRepo::update(const vector<Track> &trks, int frame, const cv::Mat &image) {
    for (auto[id, box]:trks) {
        // save normalized boxes
        box = normalize_rect(box, image.cols, image.rows);

        auto &t = targets[id];
        t.trajectories.emplace(frame, box);
        if ((frame - t.last_snap) > 5) {
            t.snapshots[frame] = image(unnormalize_rect(pad_rect(box, padding), image.cols, image.rows)).clone();
            t.last_snap = frame;
        }
    }
}

void TargetRepo::record() {
    for (auto&[id, t]:targets) {
        auto dir_path = fs::path("targets") / to_string(id);
        fs::create_directories(dir_path);

        ofstream fp(dir_path / "trajectories.txt", ios::app);
        fp << right << fixed << setprecision(3);
        for (auto &[frame, box]:t.trajectories) {
            fp << setw(6) << frame
               << setw(6) << box.x
               << setw(6) << box.y
               << setw(6) << box.width
               << setw(6) << box.height
               << setw(6) << endl;
        }
        t.trajectories.clear();

        for (auto &[frame, ss]:t.snapshots) {
            cv::imwrite((dir_path / (to_string(frame) + ".jpg")).string(), ss);
        }
        t.snapshots.clear();
    }
}
