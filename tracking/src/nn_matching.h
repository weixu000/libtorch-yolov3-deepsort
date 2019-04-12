#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include <torch/torch.h>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

torch::Tensor iou_dist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks);

#endif //NN_MATCHING_H
