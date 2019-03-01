#ifndef LETTERBOX_H
#define LETTERBOX_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

static inline std::array<int64_t, 2> letterbox_dim(torch::IntList img, torch::IntList box) {
    auto h = box[0], w = box[1];
    auto img_h = img[0], img_w = img[1];
    auto s = std::min(1.0f * w / img_w, 1.0f * h / img_h);
    return std::array{int64_t(img_h * s), int64_t(img_w * s)};
}

static inline cv::Mat letterbox_img(const cv::Mat &img, torch::IntList box) {
    auto h = box[0], w = box[1];
    auto[new_h, new_w] = letterbox_dim({img.rows, img.cols}, box);

    cv::Mat out = (cv::Mat::zeros(h, w, CV_8UC3) + 1) * 128;

    cv::resize(img,
               out({(h - new_h) / 2, (h - new_h) / 2 + new_h}, {(w - new_w) / 2, (w - new_w) / 2 + new_w}),
               {new_w, new_h},
               0, 0, cv::INTER_CUBIC);
    return out;
}

static inline void inv_letterbox_bbox(torch::Tensor bbox, torch::IntList box_dim, torch::IntList img_dim) {
    auto img_h = img_dim[0], img_w = img_dim[1];
    auto h = box_dim[0], w = box_dim[1];
    auto[new_h, new_w] = letterbox_dim(img_dim, box_dim);

    bbox.select(1, 0).add_(-(w - new_w) / 2).mul_(1.0f * img_w / new_w);
    bbox.select(1, 2).mul_(1.0f * img_w / new_w);

    bbox.select(1, 1).add_(-(h - new_h) / 2).mul_(1.0f * img_h / new_h);
    bbox.select(1, 3).mul_(1.0f * img_h / new_h);
}

#endif //LETTERBOX_H
