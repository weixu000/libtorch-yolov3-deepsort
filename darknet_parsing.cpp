//
// Created by wei-x15 on 2/27/19.
//

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

std::vector<std::map<string, string>> load_cfg(const string &cfg_file) {
    std::ifstream fs(cfg_file);
    string line;

    std::vector<std::map<string, string>> blocks;

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

