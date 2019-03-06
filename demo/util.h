#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sstream>

using ClassList=std::vector<std::string>;

inline ClassList load_classes(const std::string &path) {
    ClassList out;
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

using ColorMap=std::vector<cv::Scalar>;

inline ColorMap color_map(int64_t n) {
    auto bit_get = [](int64_t x, int64_t i) { return x & (1 << i); };
    ColorMap cmap;

    for (int64_t i = 0; i < n; ++i) {
        int64_t r = 0, g = 0, b = 0;
        int64_t i_ = i;
        for (int64_t j = 7; j >= 0; --j) {
            r |= bit_get(i_, 0) << j;
            g |= bit_get(i_, 1) << j;
            b |= bit_get(i_, 2) << j;
            i_ >>= 3;
        }
        cmap.emplace_back(b, g, r);
    }
    return cmap;
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

inline void draw_bbox(cv::Mat &img, cv::Rect2f bbox, const string &label = "", const cv::Scalar &color = {0, 0, 255}) {
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

inline void draw_trajectories(cv::Mat &img, const std::vector<std::pair<int, cv::Rect2f>> &traj,
                              float w, float h,
                              const cv::Scalar &color = {0, 0, 255}) {
    if (traj.size() < 2) return;

    auto cur = traj.begin()->second;
    auto pt1 = (cur.tl() + cur.br()) / 2;
    pt1.x *= w;
    pt1.y *= h;

    for (auto it = traj.begin() + 1; it != traj.end(); ++it) {
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

#endif //UTIL_H