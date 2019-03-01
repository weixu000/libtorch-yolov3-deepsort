#include "bbox.h"

using torch::Tensor;

torch::Tensor anchor_transform(torch::Tensor prediction,
                               torch::Tensor anchors,
                               torch::TensorList grid,
                               torch::IntList stride) {
    // sigmoid object confidence
    prediction.select(2, 4).sigmoid_();

    // softmax the class scores
    prediction.slice(2, 5) = prediction.slice(2, 5).softmax(-1);

    // sigmoid the centre_X, centre_Y
    prediction.select(2, 0).sigmoid_().add_(grid[1].view({1, 1, 1, -1})).mul_(stride[1]);
    prediction.select(2, 1).sigmoid_().add_(grid[0].view({1, 1, -1, 1})).mul_(stride[0]);

    // log space transform height and the width
    prediction.select(2, 2).exp_().mul_(anchors.select(1, 0).view({1, -1, 1, 1}));
    prediction.select(2, 3).exp_().mul_(anchors.select(1, 1).view({1, -1, 1, 1}));

    return prediction.transpose(2, -1).contiguous().view({prediction.size(0), -1, prediction.size(2)});
}

void center_to_corner(torch::Tensor bbox) {
    bbox.select(1, 0) -= bbox.select(1, 2) / 2;
    bbox.select(1, 1) -= bbox.select(1, 3) / 2;
    bbox.select(1, 2) += bbox.select(1, 0);
    bbox.select(1, 3) += bbox.select(1, 1);
}

DetTensorList threshold_confidence(torch::Tensor pred, float threshold) {
    auto max_cls_tup = pred.slice(2, 5).max(2);
    auto max_cls_score = std::get<0>(max_cls_tup);
    auto max_cls = std::get<1>(max_cls_tup);

    max_cls_score *= pred.select(2, 4);
    auto prob_thresh = max_cls_score > threshold;

    pred = pred.slice(2, 0, 4);

    DetTensorList out;
    for (int64_t i = 0; i < pred.size(0); ++i) {
        auto index = prob_thresh[i].nonzero().squeeze_();
        out.emplace_back(pred[i].index_select(0, index),
                         max_cls[i].index_select(0, index),
                         max_cls_score[i].index_select(0, index));
    }
    return out;
}

static inline float iou(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;

    return in / un;
}

void NMS(std::vector<Detection> &dets, float threshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.scr > b.scr; });

    for (size_t i = 0; i < dets.size(); ++i) {
        dets.erase(std::remove_if(dets.begin() + i + 1, dets.end(),
                                  [&](const Detection &d) {
                                      return iou(dets[i].bbox, d.bbox) > 1 - threshold;
                                  }),
                   dets.end());
    }
}

