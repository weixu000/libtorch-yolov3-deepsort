#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <sstream>

#include "bbox.h"

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

template<typename TLabelFn, typename TColorFn>
inline void draw_bbox(cv::Mat &img, torch::Tensor bbox, TLabelFn label_fn, TColorFn color_fn) {
    auto bbox_acc = bbox.accessor<float, 2>();
    for (int64_t i = 0; i < bbox_acc.size(0); ++i) {
        auto p1 = cv::Point(bbox_acc[i][0], bbox_acc[i][1]);
        auto p2 = cv::Point(bbox_acc[i][2], bbox_acc[i][3]);
        cv::rectangle(img, p1, p2, color_fn(i));

        auto label = label_fn(i);
        if (!label.empty()) {
            draw_text(img, label, color_fn(i), p1);
        }
    }
}

inline void draw_detections(cv::Mat &img, const Detection &detections,
                            const ClassList &classes, const ColorMap &cmap) {
    auto&[bbox, cls, scr] = detections;
    auto cls_acc = cls.accessor<int64_t, 1>();
    auto scr_acc = scr.accessor<float, 1>();
    std::ostringstream str;

    auto label_fn = [&](int64_t i) {
        str.str("");
        str << classes[cls_acc[i]] << " "
            << std::fixed << std::setprecision(2) << scr_acc[i];
        return str.str();
    };
    auto color_fn = [&](int64_t i) {
        return cmap[cls_acc[i]];
    };

    draw_bbox(img, bbox, label_fn, color_fn);
}

