#include "darknet_parsing.h"

using std::string;
using std::vector;
using std::map;

// trim from start (in place)
static inline void ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](char ch) {
        return !isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), [](char ch) {
        return !isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
void trim(string &s) {
    ltrim(s);
    rtrim(s);
}

int split(const string &str, vector<string> &ret_, string sep) {
    if (str.empty()) {
        return 0;
    }

    string tmp;
    string::size_type pos_begin = str.find_first_not_of(sep);
    string::size_type comma_pos = 0;

    while (pos_begin != string::npos) {
        comma_pos = str.find(sep, pos_begin);
        if (comma_pos != string::npos) {
            tmp = str.substr(pos_begin, comma_pos - pos_begin);
            pos_begin = comma_pos + sep.length();
        } else {
            tmp = str.substr(pos_begin);
            pos_begin = comma_pos;
        }

        if (!tmp.empty()) {
            trim(tmp);
            ret_.push_back(tmp);
            tmp.clear();
        }
    }
    return 0;
}

int split(const string &str, vector<int> &ret_, string sep) {
    vector<string> tmp;
    split(str, tmp, sep);

    for (int i = 0; i < tmp.size(); i++) {
        ret_.push_back(stoi(tmp[i]));
    }
}

int get_int_from_cfg(map<string, string> block, string key, int default_value) {
    if (block.find(key) != block.end()) {
        return stoi(block.at(key));
    }
    return default_value;
}

string get_string_from_cfg(map<string, string> block, string key, string default_value) {
    if (block.find(key) != block.end()) {
        return block.at(key);
    }
    return default_value;
}

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride, int64_t padding, int64_t groups, bool with_bias) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride_ = stride;
    conv_options.padding_ = padding;
    conv_options.groups_ = groups;
    conv_options.with_bias_ = with_bias;
    return conv_options;
}

torch::nn::BatchNormOptions bn_options(int64_t features) {
    torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
    bn_options.affine_ = true;
    bn_options.stateful_ = true;
    return bn_options;
}

Blocks load_cfg(const string &cfg_file) {
    std::ifstream fs(cfg_file);
    string line;

    Blocks blocks;

    if (!fs) {
        throw "Fail to load cfg file";
    }

    while (getline(fs, line)) {
        trim(line);

        if (line.empty()) {
            continue;
        }

        if (line.substr(0, 1) == "[") {
            map<string, string> block;

            string key = line.substr(1, line.length() - 2);
            block["type"] = key;

            blocks.push_back(block);
        } else {
            auto &block = blocks.back();

            vector<string> op_info;

            split(line, op_info, "=");

            if (op_info.size() == 2) {
                string p_key = op_info[0];
                string p_value = op_info[1];
                block[p_key] = p_value;
            }
        }
    }
    fs.close();

    return blocks;
}

void load_weights(const string &weight_file, const Blocks &blocks, std::vector<torch::nn::Sequential> &module_list) {
    std::ifstream fs(weight_file, std::ios_base::binary);

    // header info: 5 * int32_t
    auto header_size = sizeof(int32_t) * 5;

    int64_t index_weight = 0;

    fs.seekg(0, std::ifstream::end);
    int64_t length = fs.tellg();
    // skip header
    length = length - header_size;

    fs.seekg(header_size, std::ifstream::beg);
    auto *weights_src = (float *) malloc(static_cast<size_t>(length));
    fs.read(reinterpret_cast<char *>(weights_src), length);

    fs.close();

    at::Tensor weights = at::CPU(torch::kFloat32).tensorFromBlob(weights_src, {length / 4}, free);

    for (int i = 0; i < module_list.size(); i++) {
        auto &module_info = blocks[i + 1];
        string module_type = module_info.at("type");

        // only conv layer need to load weight
        if (module_type != "convolutional") continue;

        torch::nn::Sequential seq_module = module_list[i];

        auto conv_module = seq_module.ptr()->ptr(0);
        auto *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get());

        int batch_normalize = get_int_from_cfg(module_info, "batch_normalize", 0);

        if (batch_normalize > 0) {
            // second module
            auto bn_module = seq_module.ptr()->ptr(1);

            auto *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

            auto num_bn_biases = bn_imp->bias.numel();

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
            auto num_conv_biases = conv_imp->bias.numel();

            at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
            index_weight += num_conv_biases;

            conv_bias = conv_bias.view_as(conv_imp->bias);
            conv_imp->bias.set_data(conv_bias);
        }

        auto num_weights = conv_imp->weight.numel();

        at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
        index_weight += num_weights;

        conv_weights = conv_weights.view_as(conv_imp->weight);
        conv_imp->weight.set_data(conv_weights);
    }
}

