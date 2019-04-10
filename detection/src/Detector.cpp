#include <algorithm>

#include "Detector.h"
#include "Darknet.h"
#include "letterbox.h"

Detector::Detector(const std::array<int64_t, 2> &_inp_dim,
                   float nms, float confidence)
        : net(new Darknet("models/yolov3.cfg")),
          NMS_threshold(nms), confidence_threshold(confidence) {
    net->load_weights("weights/yolov3.weights"); // TODO: do not hard-code path
    net->to(torch::kCUDA);
    net->eval();
    torch::NoGradGuard no_grad;

    inp_dim = _inp_dim;
}

Detector::~Detector() = default;


using DetTensor = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using DetTensorList = std::vector<DetTensor>;

static inline void center_to_corner(torch::Tensor bbox) {
    bbox.select(1, 0) -= bbox.select(1, 2) / 2;
    bbox.select(1, 1) -= bbox.select(1, 3) / 2;
}

static inline DetTensor threshold_confidence(torch::Tensor pred, float threshold) {
    auto[max_cls_score, max_cls] = pred.slice(1, 5).max(1);
    max_cls_score *= pred.select(1, 4);
    auto prob_thresh = max_cls_score > threshold;

    pred = pred.slice(1, 0, 4);

    auto index = prob_thresh.nonzero().squeeze_();
    return std::make_tuple(pred.index_select(0, index),
                           max_cls.index_select(0, index),
                           max_cls_score.index_select(0, index));
}

static inline float iou(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;

    return in / un;
}

struct Detection {
    Detection(const cv::Rect2f &bbox, float scr) : bbox(bbox), scr(scr) {}

    cv::Rect2f bbox;
    float scr;
};

static inline void NMS(std::vector<Detection> &dets, float threshold) {
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

std::vector<cv::Rect2f> Detector::detect(cv::Mat image) {
    int64_t orig_dim[] = {image.rows, image.cols};
    image = letterbox_img(image, inp_dim);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    image.convertTo(image, CV_32F, 1.0 / 255);

    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(image.data,
                                                                 {1, inp_dim[0], inp_dim[1], 3})
            .permute({0, 3, 1, 2});
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(torch::kCUDA);
    auto prediction = net->forward(img_var).squeeze_(0);
    auto[bbox, cls, scr] = threshold_confidence(prediction, confidence_threshold);
    bbox = bbox.cpu();
    cls = cls.cpu();
    scr = scr.cpu();

    auto cls_mask = cls == 0;
    bbox = bbox.index_select(0, cls_mask.nonzero().squeeze_());
    scr = scr.masked_select(cls_mask);

    center_to_corner(bbox);
    inv_letterbox_bbox(bbox, inp_dim, orig_dim);

    auto bbox_acc = bbox.accessor<float, 2>();
    auto scr_acc = scr.accessor<float, 1>();
    std::vector<Detection> dets;
    for (int64_t i = 0; i < bbox_acc.size(0); ++i) {
        cv::Rect2f r(bbox_acc[i][0], bbox_acc[i][1], bbox_acc[i][2], bbox_acc[i][3]);
        dets.emplace_back(r, scr_acc[i]);
    }

    NMS(dets, NMS_threshold);

    std::vector<cv::Rect2f> out;
    for (auto &d:dets) {
        out.push_back(d.bbox);
    }

    return out;
}
