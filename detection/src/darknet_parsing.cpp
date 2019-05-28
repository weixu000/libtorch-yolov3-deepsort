#include <fstream>

#include "darknet_parsing.h"

using namespace std;

namespace {
    // trim from start (in place)
    void ltrim(string &s) {
        s.erase(s.begin(), find_if(s.begin(), s.end(), [](char ch) {
            return !isspace(ch);
        }));
    }

    // trim from end (in place)
    void rtrim(string &s) {
        s.erase(find_if(s.rbegin(), s.rend(), [](char ch) {
            return !isspace(ch);
        }).base(), s.end());
    }

    // trim from both ends (in place)
    void trim(string &s) {
        ltrim(s);
        rtrim(s);
    }

    void load_tensor(torch::Tensor t, ifstream &fs) {
        fs.read(static_cast<char *>(t.data_ptr()), t.numel() * sizeof(float));
    }
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
    return ret_.size();
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
    ifstream fs(cfg_file);
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

void load_weights(const string &weight_file, const Blocks &blocks, vector<torch::nn::Sequential> &module_list) {
    ifstream fs(weight_file, ios_base::binary);
    if (!fs) {
        throw std::runtime_error("No weight file for Darknet!");
    }

    fs.seekg(sizeof(int32_t) * 5, ios_base::beg);

    for (size_t i = 0; i < module_list.size(); i++) {
        auto &module_info = blocks[i + 1];

        // only conv layer need to load weight
        if (module_info.at("type") != "convolutional") continue;

        auto seq_module = module_list[i];

        auto conv = dynamic_pointer_cast<torch::nn::Conv2dImpl>(seq_module[0]);

        if (get_int_from_cfg(module_info, "batch_normalize", 0)) {
            // second module
            auto bn = dynamic_pointer_cast<torch::nn::BatchNormImpl>(seq_module[1]);

            load_tensor(bn->bias, fs);
            load_tensor(bn->weight, fs);
            load_tensor(bn->running_mean, fs);
            load_tensor(bn->running_var, fs);
        } else {
            load_tensor(conv->bias, fs);
        }
        load_tensor(conv->weight, fs);
    }
}

