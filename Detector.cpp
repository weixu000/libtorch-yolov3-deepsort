#include "Detector.h"

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

Detection Detector::detect(cv::Mat origin_image) {
    cv::Mat resized_image;

    cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
    cv::resize(resized_image, resized_image, cv::Size(inp_dim[1], inp_dim[0]));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data,
                                                                 {1, inp_dim[0], inp_dim[1], 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    auto img_var = torch::autograd::make_variable(img_tensor, false).to(torch::kCUDA);

    auto prediction = net.forward(img_var);

    auto out = threshold_confidence(prediction, 0.1)[0];
    auto &bbox = std::get<0>(out);
    center_to_corner(bbox);
    NMS(out, 0.4);

    auto w_scale = float(origin_image.cols) / inp_dim[0];
    auto h_scale = float(origin_image.rows) / inp_dim[1];

    bbox.select(1, 0).mul_(w_scale);
    bbox.select(1, 1).mul_(h_scale);
    bbox.select(1, 2).mul_(w_scale);
    bbox.select(1, 3).mul_(h_scale);

    return out;
}
