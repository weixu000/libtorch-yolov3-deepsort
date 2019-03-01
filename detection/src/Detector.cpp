#include "Detector.h"
#include "letterbox.h"
#include "bbox.h"

Detector::Detector(torch::IntList _inp_dim)
        : net("models/yolov3.cfg") {
    net.load_weights("weights/yolov3.weights");
    net.to(torch::kCUDA);
    net.eval();
    torch::NoGradGuard no_grad;

    if (_inp_dim.empty()) {
        inp_dim[0] = std::atoi(net.get_net_info().at("height").c_str());
        inp_dim[1] = std::atoi(net.get_net_info().at("width").c_str());
    } else {
        inp_dim[0] = _inp_dim[0];
        inp_dim[1] = _inp_dim[1];
    }
}

Detection Detector::detect(cv::Mat image) {
    int64_t orig_dim[] = {image.rows, image.cols};
    image = letterbox_img(image, inp_dim);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    image.convertTo(image, CV_32F, 1.0 / 255);

    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(image.data,
                                                                 {1, inp_dim[0], inp_dim[1], 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(torch::kCUDA);

    auto prediction = net.forward(img_var);

    auto out = threshold_confidence(prediction, 0.1)[0];
    auto &[bbox, cls, scr] = out;
    bbox = bbox.cpu();
    cls = cls.cpu();
    scr = scr.cpu();

    center_to_corner(bbox);
    NMS(out, 0.4);

    inv_letterbox_bbox(bbox, inp_dim, orig_dim);

    return out;
}
