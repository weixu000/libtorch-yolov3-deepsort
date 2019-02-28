/*******************************************************************************
* 
* Author : walktree
* Email  : walktree@gmail.com
*
* A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. 
* It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~
*
*******************************************************************************/
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

    torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device) {
        return anchor_transform(prediction, {inp_dim, inp_dim}, anchors, num_classes);
    }
};

Darknet::Darknet(const char *cfg_file, torch::Device *device) {
    blocks = load_cfg(cfg_file);

    _device = device;

    create_modules();
}

map<string, string> *Darknet::get_net_info() {
    if (blocks.size() > 0) {
        return &blocks[0];
    }
}

void Darknet::load_weights(const char *weight_file) {
    ifstream fs(weight_file, ios_base::binary);

    // header info: 5 * int32_t
    int32_t header_size = sizeof(int32_t) * 5;

    int64_t index_weight = 0;

    fs.seekg(0, fs.end);
    int64_t length = fs.tellg();
    // skip header
    length = length - header_size;

    fs.seekg(header_size, fs.beg);
    float *weights_src = (float *) malloc(length);
    fs.read(reinterpret_cast<char *>(weights_src), length);

    fs.close();

    at::TensorOptions options = at::TensorOptions()
            .dtype(torch::kFloat32)
            .is_variable(true);
    at::Tensor weights = at::CPU(torch::kFloat32).tensorFromBlob(weights_src, {length / 4});

    for (int i = 0; i < module_list.size(); i++) {
        map<string, string> module_info = blocks[i + 1];

        string module_type = module_info["type"];

        // only conv layer need to load weight
        if (module_type != "convolutional") continue;

        torch::nn::Sequential seq_module = module_list[i];

        auto conv_module = seq_module.ptr()->ptr(0);
        torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get());

        int batch_normalize = get_int_from_cfg(module_info, "batch_normalize", 0);

        if (batch_normalize > 0) {
            // second module
            auto bn_module = seq_module.ptr()->ptr(1);

            torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

            int num_bn_biases = bn_imp->bias.numel();

            at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
            index_weight += num_bn_biases;

            at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
            index_weight += num_bn_biases;

            at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
            index_weight += num_bn_biases;

            at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
            index_weight += num_bn_biases;

            bn_bias = bn_bias.view_as(bn_imp->bias);
            bn_weights = bn_weights.view_as(bn_imp->weight);
            bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
            bn_running_var = bn_running_var.view_as(bn_imp->running_variance);

            bn_imp->bias.set_data(bn_bias);
            bn_imp->weight.set_data(bn_weights);
            bn_imp->running_mean.set_data(bn_running_mean);
            bn_imp->running_variance.set_data(bn_running_var);
        } else {
            int num_conv_biases = conv_imp->bias.numel();

            at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
            index_weight += num_conv_biases;

            conv_bias = conv_bias.view_as(conv_imp->bias);
            conv_imp->bias.set_data(conv_bias);
        }

        int num_weights = conv_imp->weight.numel();

        at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
        index_weight += num_weights;

        conv_weights = conv_weights.view_as(conv_imp->weight);
        conv_imp->weight.set_data(conv_weights);
    }
}

torch::Tensor Darknet::forward(torch::Tensor x) {
    int module_count = module_list.size();

    std::vector<torch::Tensor> outputs(module_count);

    torch::Tensor result;
    int write = 0;

    for (int i = 0; i < module_count; i++) {
        map<string, string> block = blocks[i + 1];

        string layer_type = block["type"];

        if (layer_type == "net")
            continue;

        if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool") {
            torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());

            x = seq_imp->forward(x);
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
            torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());

            map<string, string> net_info = blocks[0];
            int inp_dim = get_int_from_cfg(net_info, "height", 0);
            int num_classes = get_int_from_cfg(block, "classes", 0);

            x = seq_imp->forward(x, inp_dim, num_classes, *_device);

            if (write == 0) {
                result = x;
                write = 1;
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

    for (int i = 0, len = blocks.size(); i < len; i++) {
        map<string, string> block = blocks[i];

        string layer_type = block["type"];

        // std::cout << index << "--" << layer_type << endl;

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

            blocks[i] = block;

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

            blocks[i] = block;

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
