#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>

inline cv::Rect2f pad_rect(cv::Rect2f rect, float padding) {
    rect.x = std::max(0.0f, rect.x - rect.width * padding);
    rect.y = std::max(0.0f, rect.y - rect.height * padding);
    rect.width = std::min(1 - rect.x, rect.width * (1 + 2 * padding));
    rect.height = std::min(1 - rect.y, rect.height * (1 + 2 * padding));

    return rect;
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