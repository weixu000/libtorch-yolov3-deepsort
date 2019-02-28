#include "Darknet.h"
#include "darknet_parsing.h"


struct EmptyLayer : torch::nn::Module {
    EmptyLayer() {

    }

    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};

struct UpsampleLayer : torch::nn::Module {
    int _stride;

    UpsampleLayer(int stride) {
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {

        torch::IntList sizes = x.sizes();

        int64_t w, h;

        if (sizes.size() == 4) {
            w = sizes[2] * _stride;
            h = sizes[3] * _stride;

            x = torch::upsample_nearest2d(x, {w, h});
        } else if (sizes.size() == 3) {
            w = sizes[2] * _stride;
            x = torch::upsample_nearest1d(x, {w});
        }
        return x;
    }
};

struct MaxPoolLayer2D : torch::nn::Module {
    int _kernel_size;
    int _stride;

    MaxPoolLayer2D(int kernel_size, int stride) {
        _kernel_size = kernel_size;
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {
        if (_stride != 1) {
            x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
        } else {
            int pad = _kernel_size - 1;

            torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
            x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
        }

        return x;
    }
};

struct DetectionLayer : torch::nn::Module {
    vector<float> anchors;

    DetectionLayer(vector<float> _anchors) {
        anchors = _anchors;
//        anchors = torch::from_blob(_anchors.data(),
//                                   {static_cast<int64_t>(_anchors.size() / 2), 2});
    }

    torch::Tensor forward(torch::Tensor prediction, torch::IntList inp_dim, int num_classes) {
        return anchor_transform(prediction, inp_dim, anchors, num_classes);
    }
};

Darknet::Darknet(const char *cfg_file) {
    blocks = load_cfg(cfg_file);

    create_modules();
}

void Darknet::load_weights(const char *weight_file) {
    ::load_weights(weight_file, blocks, module_list);
}

torch::Tensor Darknet::forward(torch::Tensor x) {
    auto inp_dim = x.sizes().slice(2);
    auto module_count = module_list.size();

    std::vector<torch::Tensor> outputs(module_count);

    torch::Tensor result;
    auto write = false;

    for (int i = 0; i < module_count; i++) {
        auto block = blocks[i + 1];

        auto layer_type = block["type"];

        if (layer_type == "net")
            continue;

        if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool") {
            x = module_list[i]->forward(x);
            outputs[i] = x;
        } else if (layer_type == "route") {
            int start = std::stoi(block["start"]);
            int end = std::stoi(block["end"]);

            if (start > 0) start = start - i;

            if (end == 0) {
                x = outputs[i + start];
            } else {
                if (end > 0) end = end - i;

                torch::Tensor map_1 = outputs[i + start];
                torch::Tensor map_2 = outputs[i + end];

                x = torch::cat({map_1, map_2}, 1);
            }

            outputs[i] = x;
        } else if (layer_type == "shortcut") {
            int from = std::stoi(block["from"]);
            x = outputs[i - 1] + outputs[i + from];
            outputs[i] = x;
        } else if (layer_type == "yolo") {
            auto net_info = blocks[0];
            int num_classes = get_int_from_cfg(block, "classes", 0);

            x = module_list[i]->forward(x, inp_dim, num_classes);

            if (!write) {
                result = x;
                write = true;
            } else {
                result = torch::cat({result, x}, 1);
            }

            outputs[i] = outputs[i - 1];
        }
    }
    return result;
}

void Darknet::create_modules() {
    int prev_filters = 3;

    vector<int> output_filters;

    int index = 0;

    int filters = 0;

    for (auto &block:blocks) {
        auto layer_type = block["type"];

        torch::nn::Sequential module;

        if (layer_type == "net")
            continue;

        if (layer_type == "convolutional") {
            string activation = get_string_from_cfg(block, "activation", "");
            int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
            filters = get_int_from_cfg(block, "filters", 0);
            int padding = get_int_from_cfg(block, "pad", 0);
            int kernel_size = get_int_from_cfg(block, "size", 0);
            int stride = get_int_from_cfg(block, "stride", 1);

            int pad = padding > 0 ? (kernel_size - 1) / 2 : 0;
            bool with_bias = batch_normalize > 0 ? false : true;

            torch::nn::Conv2d conv = torch::nn::Conv2d(
                    conv_options(prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
            module->push_back(conv);

            if (batch_normalize > 0) {
                torch::nn::BatchNorm bn = torch::nn::BatchNorm(bn_options(filters));
                module->push_back(bn);
            }

            if (activation == "leaky") {
                module->push_back(torch::nn::Functional(at::leaky_relu, /*slope=*/0.1));
            }
        } else if (layer_type == "upsample") {
            int stride = get_int_from_cfg(block, "stride", 1);

            UpsampleLayer uplayer(stride);
            module->push_back(uplayer);
        } else if (layer_type == "maxpool") {
            int stride = get_int_from_cfg(block, "stride", 1);
            int size = get_int_from_cfg(block, "size", 1);

            MaxPoolLayer2D poolLayer(size, stride);
            module->push_back(poolLayer);
        } else if (layer_type == "shortcut") {
            // skip connection
            int from = get_int_from_cfg(block, "from", 0);
            block["from"] = to_string(from);

            // placeholder
            EmptyLayer layer;
            module->push_back(layer);
        } else if (layer_type == "route") {
            // L 85: -1, 61
            string layers_info = get_string_from_cfg(block, "layers", "");

            vector<string> layers;
            split(layers_info, layers, ",");

            string::size_type sz;
            signed int start = stoi(layers[0], &sz);
            signed int end = 0;

            if (layers.size() > 1) {
                end = stoi(layers[1], &sz);
            }

            if (start > 0) start = start - index;

            if (end > 0) end = end - index;

            block["start"] = to_string(start);
            block["end"] = to_string(end);

            // placeholder
            EmptyLayer layer;
            module->push_back(layer);

            if (end < 0) {
                filters = output_filters[index + start] + output_filters[index + end];
            } else {
                filters = output_filters[index + start];
            }
        } else if (layer_type == "yolo") {
            string mask_info = get_string_from_cfg(block, "mask", "");
            vector<int> masks;
            split(mask_info, masks, ",");

            string anchor_info = get_string_from_cfg(block, "anchors", "");
            vector<int> anchors;
            split(anchor_info, anchors, ",");

            vector<float> anchor_points;
            int pos;
            for (int i = 0; i < masks.size(); i++) {
                pos = masks[i];
                anchor_points.push_back(anchors[pos * 2]);
                anchor_points.push_back(anchors[pos * 2 + 1]);
            }

            DetectionLayer layer(anchor_points);
            module->push_back(layer);
        } else {
            cout << "unsupported operator:" << layer_type << endl;
        }

        prev_filters = filters;
        output_filters.push_back(filters);
        module_list.push_back(module);

        char *module_key = new char[strlen("layer_") + sizeof(index) + 1];

        sprintf(module_key, "%s%d", "layer_", index);

        register_module(module_key, module);

        index += 1;
    }
}

Detection write_results(torch::Tensor prediction, float confidence, float nms_conf) {
    auto out = threshold_confidence(prediction, confidence)[0];
    center_to_corner(std::get<0>(out));
    NMS(out, nms_conf);

    return out;
}
