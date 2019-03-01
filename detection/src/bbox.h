#ifndef BBOX_H
#define BBOX_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using DetTensor = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using DetTensorList = std::vector<DetTensor>;

torch::Tensor anchor_transform(torch::Tensor prediction,
                               torch::Tensor anchors,
                               torch::TensorList grid,
                               torch::IntList stride);


void center_to_corner(torch::Tensor bbox);

DetTensorList threshold_confidence(torch::Tensor pred, float threshold);

struct Detection {
    cv::Rect2f bbox;
    float scr;
};

void NMS(std::vector<Detection> &dets, float threshold);

#endif //BBOX_H
