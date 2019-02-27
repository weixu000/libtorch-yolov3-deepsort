#include "bbox.h"

using torch::Tensor;

// returns the IoU of two bounding boxes
Tensor iou(Tensor box1, Tensor box2) {
    auto b1_x1 = box1.select(1, 0).unsqueeze_(1);
    auto b1_y1 = box1.select(1, 1).unsqueeze_(1);
    auto b1_x2 = box1.select(1, 2).unsqueeze_(1);
    auto b1_y2 = box1.select(1, 3).unsqueeze_(1);

    auto b2_x1 = box2.select(1, 0).unsqueeze_(0);
    auto b2_y1 = box2.select(1, 1).unsqueeze_(0);
    auto b2_x2 = box2.select(1, 2).unsqueeze_(0);
    auto b2_y2 = box2.select(1, 3).unsqueeze_(0);

    auto inter_x1 = max(b1_x1, b2_x1);
    auto inter_y1 = max(b1_y1, b2_y1);
    auto inter_x2 = min(b1_x2, b2_x2);
    auto inter_y2 = min(b1_y2, b2_y2);

    auto inter_area = clamp(inter_x2 - inter_x1 + 1, 0) * clamp(inter_y2 - inter_y1 + 1, 0);

    auto b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
    auto b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

    return inter_area / (b1_area + b2_area - inter_area);
}

torch::Tensor anchor_transform(torch::Tensor prediction,
                               torch::IntList inp_dim,
                               std::vector<float> _anchors,
                               int num_classes) {
    auto anchors = torch::from_blob(_anchors.data(),
                                    {static_cast<int64_t>(_anchors.size() / 2), 2}).to(prediction.device());

    auto batch_size = prediction.size(0);
    auto grid_size = prediction.sizes().slice(2);
    int64_t stride[] = {inp_dim[0] / grid_size[0], inp_dim[1] / grid_size[1]};
    auto bbox_attrs = 5 + num_classes;
    auto num_anchors = anchors.size(0);

    prediction = prediction.view({batch_size, num_anchors, bbox_attrs, grid_size[0], grid_size[1]});

    // sigmoid object confidence
    prediction.select(2, 4).sigmoid_();

    // softmax the class scores
    prediction.slice(2, 5) = prediction.slice(2, 5).softmax(-1);

    // sigmoid the centre_X, centre_Y
    auto grid = torch::arange(grid_size[1],
                              torch::dtype(prediction.dtype()).device(prediction.device()));
    prediction.select(2, 0).sigmoid_().add_(grid.view({1, 1, 1, -1})).mul_(stride[1]);
    grid = torch::arange(grid_size[0],
                         torch::dtype(prediction.dtype()).device(prediction.device()));
    prediction.select(2, 1).sigmoid_().add_(grid.view({1, 1, -1, 1})).mul_(stride[0]);

    // log space transform height and the width
    prediction.select(2, 2).exp_().mul_(anchors.select(1, 0).view({1, -1, 1, 1}));
    prediction.select(2, 3).exp_().mul_(anchors.select(1, 1).view({1, -1, 1, 1}));

    return prediction.transpose(2, -1).contiguous().view({batch_size, -1, bbox_attrs});
}

void center_to_corner(torch::Tensor &bbox) {
    bbox.select(1, 0) -= bbox.select(1, 2) / 2;
    bbox.select(1, 1) -= bbox.select(1, 3) / 2;
    bbox.select(1, 2) += bbox.select(1, 0);
    bbox.select(1, 3) += bbox.select(1, 1);
}

DetectionList threshold_confidence(torch::Tensor pred, float threshold) {
    auto max_cls_tup = pred.slice(2, 5).max(2);
    auto max_cls_score = std::get<0>(max_cls_tup);
    auto max_cls = std::get<1>(max_cls_tup);

    max_cls_score *= pred.select(2, 4);
    auto prob_thresh = max_cls_score > threshold;

    pred = pred.slice(2, 0, 4);

    DetectionList out;
    for (int64_t i = 0; i < pred.size(0); ++i) {
        auto index = prob_thresh[i].nonzero().squeeze_();
        out.emplace_back(pred[i].index_select(0, index).cpu(),
                         max_cls[i].index_select(0, index).cpu(),
                         max_cls_score[i].index_select(0, index).cpu());
    }
    return out;
}

void NMS(Detection &batch, float threshold) {
    auto bbox_batch = std::get<0>(batch);
    auto cls_batch = std::get<1>(batch);
    auto scr_batch = std::get<2>(batch);

    std::list<int64_t> cls_unique;
    auto cls_batch_acc = cls_batch.accessor<int64_t, 1>();
    for (int64_t i = 0; i < cls_batch_acc.size(0); ++i) {
        cls_unique.emplace_back(cls_batch_acc[i]);
    }
    cls_unique.sort();
    cls_unique.unique();

    auto ind_batch = torch::empty({0}, torch::dtype(torch::kInt64));
    for (auto &cls:cls_unique) {
        auto ind_cls = (cls_batch == cls).nonzero().squeeze(1);
        auto bbox_cls = bbox_batch.index_select(0, ind_cls);
        auto scr_cls = scr_batch.index_select(0, ind_cls);

        auto sort_tup = scr_cls.sort(-1, true);
        scr_cls = std::get<0>(sort_tup);
        auto sorted_ind = std::get<1>(sort_tup);
        ind_cls = ind_cls.index_select(0, sorted_ind);
        bbox_cls = bbox_cls.index_select(0, sorted_ind);

        int64_t i = 0;
        while (i < ind_cls.size(0)) {
            auto ious = iou(bbox_cls[i].unsqueeze(0), bbox_cls.slice(0, i + 1)).squeeze(0);

            auto iou_ind = (ious < threshold).nonzero().squeeze(1);

            ind_cls.slice(0, i + 1, i + 1 + iou_ind.size(0)) = ind_cls.index_select(0, i + 1 + iou_ind);
            bbox_cls.slice(0, i + 1, i + 1 + iou_ind.size(0)) = bbox_cls.index_select(0, i + 1 + iou_ind);

            ind_cls = ind_cls.slice(0, 0, i + 1 + iou_ind.size(0));
            bbox_cls = bbox_cls.slice(0, 0, i + 1 + iou_ind.size(0));

            ++i;
        }
        ind_batch = torch::cat({ind_batch, ind_cls});
    }
    std::get<0>(batch) = std::get<0>(batch).index_select(0, ind_batch);
    std::get<1>(batch) = std::get<1>(batch).index_select(0, ind_batch);
    std::get<2>(batch) = std::get<2>(batch).index_select(0, ind_batch);
}


