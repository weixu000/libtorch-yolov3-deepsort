#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <algorithm>
#include "imgui/imgui.h"
#include "Detector.h"
#include "Tracker.h"

inline auto load_classes(const std::string &path) {
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

inline cv::Scalar color_map(int64_t n) {
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

inline void draw_text(cv::Mat &img, const std::string &str,
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

inline void draw_bbox(cv::Mat &img, cv::Rect2f bbox, const string &label = "", const cv::Scalar &color = {0, 0, 0}) {
    cv::rectangle(img, bbox, color);
    if (!label.empty()) {
        draw_text(img, label, color, bbox.tl());
    }
}

inline void mat_to_texture(const cv::Mat &mat, GLuint texture) {
    assert(mat.isContinuous());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB,
                 mat.cols, mat.rows,
                 0,
                 GL_BGR, GL_UNSIGNED_BYTE, mat.data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

inline void draw_trajectories(cv::Mat &img, const std::map<int, cv::Rect2f> &traj,
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

inline cv::Rect2f normalize_rect(cv::Rect2f rect, float w, float h) {
    rect.x /= w;
    rect.y /= h;
    rect.width /= w;
    rect.height /= h;
    return rect;
}

inline cv::Rect2f unnormalize_rect(cv::Rect2f rect, float w, float h) {
    rect.x *= w;
    rect.y *= h;
    rect.width *= w;
    rect.height *= h;
    return rect;
}

inline ImVec2 image_window(const char *name, GLuint texture, bool *p_open = __null) {
    ImGui::Begin(name, p_open);
    ImGui::Image(reinterpret_cast<ImTextureID>(texture), ImGui::GetContentRegionAvail());
    ImGui::End();
    return ImGui::GetContentRegionAvail(); // return size for image uploading
}

inline void draw_dets_window(const cv::Mat &image, const std::vector<cv::Rect2f> &dets,
                             GLuint tex, bool *p_open = __null) {
    auto size = image_window("Detection", tex, p_open);
    cv::Mat dets_image;
    resize(image, dets_image, {size[0], size[1]});
    for (auto &d:dets) {
        draw_bbox(dets_image, unnormalize_rect(d, size[0], size[1]));
    }
    mat_to_texture(dets_image, tex);
}

inline void draw_trks_window(const cv::Mat &image, const std::vector<Track> &trks,
                             GLuint tex, bool *p_open = __null) {
    auto size = image_window("Tracking", tex, p_open);
    cv::Mat trks_image;
    resize(image, trks_image, {size[0], size[1]});
    for (auto &t:trks) {
        draw_bbox(trks_image, unnormalize_rect(t.box, size[0], size[1]), std::to_string(t.id));
    }
    mat_to_texture(trks_image, tex);
}

#endif //UTIL_H