#ifndef DETECTOR_H
#define DETECTOR_H

#include "Darknet.h"
#include <opencv2/opencv.hpp>

using Detection = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

class Detector {
public:
    explicit Detector(torch::IntList _inp_dim = torch::IntList());

    Detection detect(cv::Mat image);

private:
    Darknet net;
    int64_t inp_dim[2];
};


#endif //DETECTOR_H
