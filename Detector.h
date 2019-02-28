#ifndef DETECTOR_H
#define DETECTOR_H

#include "Darknet.h"
#include <opencv2/opencv.hpp>

class Detector {
public:
    explicit Detector(torch::IntList _inp_dim = torch::IntList());

    Detection detect(cv::Mat origin_image);

private:
    Darknet net;
    int64_t inp_dim[2];
};


#endif //DETECTOR_H
