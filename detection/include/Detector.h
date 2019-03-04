#ifndef DETECTOR_H
#define DETECTOR_H

#include "Darknet.h"
#include <opencv2/opencv.hpp>

class Detector {
public:
    explicit Detector(torch::IntList _inp_dim = torch::IntList());

    std::vector<cv::Rect2f> detect(cv::Mat image);

private:
    Darknet net; // TODO: maybe hide Darknet
    int64_t inp_dim[2];
};


#endif //DETECTOR_H
