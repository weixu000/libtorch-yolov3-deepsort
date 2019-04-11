#include "Darknet.h"
#include "darknet_parsing.h"

using namespace std;

struct EmptyLayerImpl : torch::nn::Module {
    EmptyLayerImpl() = default;

    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};

TORCH_MODULE(EmptyLayer);

struct UpsampleLayerImpl : torch::nn::Module {
    int _stride;

    explicit UpsampleLayerImpl(int stride) : _stride(stride) {}

    torch::Tensor forward(torch::Tensor x) {
        auto sizes = x.sizes();

        auto w = sizes[2] * _stride;
        auto h = sizes[3] * _stride;

        return torch::upsample_nearest2d(x, {w, h});
    }
};

TORCH_MODULE(UpsampleLayer);

struct MaxPoolLayer2DImpl : torch::nn::Module {
    int _kernel_size;
    int _stride;

    MaxPoolLayer2DImpl(int kernel_size, int stride) : _kernel_size(kernel_size), _stride(stride) {}

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

TORCH_MODULE(MaxPoolLayer2D);

struct DetectionLayerImpl : torch::nn::Module {
    torch::Tensor anchors;
    std::array<torch::Tensor, 2> grid;

    explicit DetectionLayerImpl(const ::std::vector<float> &_anchors)
            : anchors(register_buffer("anchors",
                                      torch::from_blob((void *) _anchors.data(),
                                                       {static_cast<int64_t>(_anchors.size() / 2), 2}).clone())),
              grid{torch::empty({0}), torch::empty({0})} {}

    torch::Tensor forward(torch::Tensor prediction, torch::IntList inp_dim) {
        auto grid_size = prediction.sizes().slice(2);
        if (grid_size[0] != grid[0].size(0) || grid_size[1] != grid[1].size(0)) {
            // update grid if size not match
            grid = {torch::arange(grid_size[0], prediction.options()),
                    torch::arange(grid_size[1], prediction.options())};
        }

        auto batch_size = prediction.size(0);
        int64_t stride[] = {inp_dim[0] / grid_size[0], inp_dim[1] / grid_size[1]};
        auto num_anchors = anchors.size(0);
        auto bbox_attrs = prediction.size(1) / num_anchors;
        prediction = prediction.view({batch_size, num_anchors, bbox_attrs, grid_size[0], grid_size[1]});

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
};

TORCH_MODULE(DetectionLayer);


Detector::Darknet::Darknet(const string &cfg_file) {
    blocks = load_cfg(cfg_file);

    create_modules();
}

void Detector::Darknet::load_weights(const string &weight_file) {
    ::load_weights(weight_file, blocks, module_list); // TODO: remove this function
}

// TODO: reimplement the python version
torch::Tensor Detector::Darknet::forward(torch::Tensor x) {
    auto inp_dim = x.sizes().slice(2);
    auto module_count = module_list.size();

    std::vector<torch::Tensor> outputs(module_count);

    vector<torch::Tensor> result;

    for (int i = 0; i < module_count; i++) {
        auto block = blocks[i + 1];

        auto layer_type = block["type"];

        if (layer_type == "net")
            continue;
        else if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool") {
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
            x = module_list[i]->forward(x, inp_dim);
            result.push_back(x);
            outputs[i] = outputs[i - 1];
        }
    }
    return torch::cat(result, 1);
}

// TODO: reimplement the python version
void Detector::Darknet::create_modules() {
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
            bool with_bias = batch_normalize <= 0;

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
            for (auto mask : masks) {
                anchor_points.push_back(anchors[mask * 2]);
                anchor_points.push_back(anchors[mask * 2 + 1]);
            }

            DetectionLayer layer(anchor_points);
            module->push_back(layer);
        } else {
            cout << "unsupported operator:" << layer_type << endl;
        }

        prev_filters = filters;
        output_filters.push_back(filters);
        module_list.push_back(module);

        register_module("layer_" + to_string(index), module);

        index += 1;
    }
}
