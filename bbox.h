#ifndef BBOX_H
#define BBOX_H

#include <torch/torch.h>

torch::Tensor iou(torch::Tensor box1, torch::Tensor box2);

torch::Tensor anchor_transform(torch::Tensor prediction,
                               torch::Tensor anchors,
                               torch::TensorList grid,
                               torch::IntList stride);

using Detection = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using DetectionList = std::vector<Detection>;

void center_to_corner(torch::Tensor bbox);

DetectionList threshold_confidence(torch::Tensor pred, float threshold);

void NMS(Detection &batch, float threshold);

#endif //BBOX_H
