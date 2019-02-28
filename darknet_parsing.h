#ifndef DARKNET_PARSING_H
#define DARKNET_PARSING_H

#include <string>
#include <map>
#include <torch/torch.h>

void trim(std::string &s);

int split(const string &str, std::vector<string> &ret_, string sep = ",");

int split(const string &str, std::vector<int> &ret_, string sep = ",");

int get_int_from_cfg(std::map<std::string, std::string> block, std::string key, int default_value);

std::string get_string_from_cfg(std::map<string, string> block, std::string key, std::string default_value);

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride, int64_t padding, int64_t groups, bool with_bias = false);

torch::nn::BatchNormOptions bn_options(int64_t features);

std::vector<std::map<string, string>> load_cfg(const string &cfg_file);

#endif //DARKNET_PARSING_H
