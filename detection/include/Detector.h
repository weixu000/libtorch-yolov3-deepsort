#ifndef DETECTOR_H
#define DETECTOR_H

#include <memory>
#include <array>
#include <opencv2/opencv.hpp>

class Darknet;

class Detector {
public:
    explicit Detector(const std::array<int64_t, 2> &_inp_dim,
                      float nms = 0.5f, float confidence = 0.6f);

    ~Detector();

    std::vector<cv::Rect2f> detect(cv::Mat image);

private:
    std::unique_ptr<Darknet> net;

    std::array<int64_t, 2> inp_dim;
    const float NMS_threshold;
    const float confidence_threshold;
};


#endif //DETECTOR_H
