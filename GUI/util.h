#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>
#include <sstream>
#include <algorithm>

#include "imgui/imgui.h"
#include "TargetRepo.h"

namespace {
    auto load_classes(const std::string &path) {
        std::vector<std::string> out;
        std::ifstream fp(path);
        while (fp) {
            std::string cls;
            std::getline(fp, cls);
            if (!cls.empty()) {
                out.push_back(cls);
            }
        }
        fp.close();
        return out;
    }

    cv::Scalar color_map(int64_t n) {
        auto bit_get = [](int64_t x, int64_t i) { return x & (1 << i); };

        int64_t r = 0, g = 0, b = 0;
        int64_t i = n;
        for (int64_t j = 7; j >= 0; --j) {
            r |= bit_get(i, 0) << j;
            g |= bit_get(i, 1) << j;
            b |= bit_get(i, 2) << j;
            i >>= 3;
        }
        return cv::Scalar(b, g, r);
    }

    void draw_text(cv::Mat &img, const std::string &str,
                   const cv::Scalar &color, cv::Point pos, bool reverse = false) {
        auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, nullptr);
        cv::Point bottom_left, upper_right;
        if (reverse) {
            upper_right = pos;
            bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
        } else {
            bottom_left = pos;
            upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
        }

        cv::rectangle(img, bottom_left, upper_right, color, -1);
        cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255) - color);
    }

    void draw_bbox(cv::Mat &img, cv::Rect2f bbox,
                   const std::string &label = "", const cv::Scalar &color = {0, 0, 0}) {
        cv::rectangle(img, bbox, color);
        if (!label.empty()) {
            draw_text(img, label, color, bbox.tl());
        }
    }


    void draw_trajectories(cv::Mat &img, const std::map<int, cv::Rect2f> &traj,
                           float w, float h,
                           const cv::Scalar &color = {0, 0, 0}) {
        if (traj.size() < 2) return;

        auto cur = traj.begin()->second;
        auto pt1 = (cur.tl() + cur.br()) / 2;
        pt1.x *= w;
        pt1.y *= h;

        for (auto it = ++traj.begin(); it != traj.end(); ++it) {
            cur = it->second;
            auto pt2 = (cur.tl() + cur.br()) / 2;
            pt2.x *= w;
            pt2.y *= h;
            cv::line(img, pt1, pt2, color);
            pt1 = pt2;
        }
    }

    cv::Rect2f unnormalize_rect(cv::Rect2f rect, float w, float h) {
        rect.x *= w;
        rect.y *= h;
        rect.width *= w;
        rect.height *= h;
        return rect;
    }

    ImVec2 image_window(const char *name, GLuint texture, bool *p_open = __null) {
        ImGui::Begin(name, p_open);
        ImGui::Image(reinterpret_cast<ImTextureID>(texture), ImGui::GetContentRegionAvail());
        ImGui::End();
        return ImGui::GetContentRegionAvail(); // return size for image uploading
    }

    void draw_res_window(const cv::Mat &image, TargetRepo &repo,
                         uint32_t display_frame, int hovered, bool *p_open = __null) {
        static auto tex = cv::ogl::Texture2D();

        auto size = image_window("Result", tex.texId(), p_open);
        auto ret_image = cv::Mat();
        cv::resize(image, ret_image, {int(size[0]), int(size[1])});
        for (std::size_t i = 0; i < repo.size(); ++i) {
            auto &t = repo[i];
            if (t.trajectories.count(display_frame)) {
                auto color = hovered == -1 ? color_map(i) : hovered == i ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 0);
                draw_trajectories(ret_image, t.trajectories, size[0], size[1], color);
                draw_bbox(ret_image, unnormalize_rect(t.trajectories.at(display_frame), size[0], size[1]),
                          std::to_string(i), color);
            }
        }
        tex.copyFrom(ret_image, true);
    }

    auto draw_target_window(TargetRepo &repo, int FPS, bool *p_open = __null) {
        using namespace std::chrono;

        int hovered = -1;
        int rewind = -1;

        static int hovered_prev = -1;
        static steady_clock::time_point hovered_start;

        ImGui::Begin("Targets", p_open);
        auto &style = ImGui::GetStyle();
        auto window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        for (std::size_t i = 0; i < repo.size(); ++i) {
            auto &t = repo[i];
            if (t.snapshots.empty() || t.trajectories.empty()) continue;
            ImGui::PushID(i);
            {
                ImGui::BeginGroup();
                {
                    auto it = t.snapshots.begin();
                    if (hovered_prev == i) {
                        auto interval = 5 * 1000 / FPS;
                        for (auto duration = duration_cast<milliseconds>(steady_clock::now() - hovered_start).count();
                             duration > interval; duration -= interval) {
                            // repeatedly play snapshots
                            if (++it == t.snapshots.end()) {
                                it = t.snapshots.begin();
                            }
                        }
                    }
                    ImGui::Image(reinterpret_cast<void *>(it->second.texId()), {50, 50});
                    if (ImGui::IsItemHovered()) {
                        hovered = i;
                    }
                }
                ImGui::SameLine();
                {
                    ImGui::BeginGroup();
                    ImGui::Text("Id: %zu", i);
                    ImGui::Text("Duration: %3d, %3d", t.trajectories.begin()->first, t.trajectories.rbegin()->first);
                    if (ImGui::Button("Rewind")) {
                        rewind = t.trajectories.begin()->first;
                    }
                    ImGui::EndGroup();
                }
                ImGui::EndGroup();
            }
            ImGui::PopID();

            auto last_x2 = ImGui::GetItemRectMax().x;
            auto next_x2 = last_x2 + style.ItemSpacing.x
                           + ImGui::GetItemRectSize().x; // Expected position if next button was on same line
            if (i != repo.size() - 1 && next_x2 < window_visible_x2)
                ImGui::SameLine();
        }
        ImGui::End();

        if (hovered != hovered_prev) {
            hovered_prev = hovered;
            hovered_start = steady_clock::now();
        }

        return std::make_pair(hovered, rewind);
    }

    void draw_control_window(cv::VideoCapture &cap, uint32_t &processed_frame, bool &show_demo_window,
                             bool &show_res_window, bool &show_target_window, bool &playing, bool &next) {
        next = false;
        uint32_t frame_min = 0, frame_max = static_cast<uint32_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        ImGui::Begin("Control", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
        ImGui::Text("GUI Framerate: %d", static_cast<int>(ImGui::GetIO().Framerate));
        ImGui::Text("Video Framerate: %d", static_cast<int>(cap.get(cv::CAP_PROP_FPS)));
        ImGui::Separator();
        ImGui::Checkbox("Show demo window", &show_demo_window);
        ImGui::Checkbox("Show result window", &show_res_window);
        ImGui::Checkbox("Show target window", &show_target_window);
        ImGui::Separator();
        ImGui::Checkbox("Playing", &playing);
        if (auto display_frame = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
                ImGui::SliderScalar("", ImGuiDataType_U32, &display_frame, &frame_min, &processed_frame,
                                    ("%u/" + std::to_string(processed_frame)).c_str())) {
            cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(display_frame));
            next = true;
        }
        next |= ImGui::Button("Next");
        ImGui::End();
    }
}

#endif //UTIL_H