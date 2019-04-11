#ifndef DARKNET_PARSING_H
#define DARKNET_PARSING_H

#include <string>
#include <map>
#include <torch/torch.h>

int split(const std::string &str, std::vector<std::string> &ret_, std::string sep = ",");

int split(const std::string &str, std::vector<int> &ret_, std::string sep = ",");

int get_int_from_cfg(std::map<std::string, std::string> block, std::string key, int default_value);

std::string get_string_from_cfg(std::map<std::string, std::string> block, std::string key, std::string default_value);

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride, int64_t padding, int64_t groups, bool with_bias = false);

torch::nn::BatchNormOptions bn_options(int64_t features);

using Blocks = std::vector<std::map<std::string, std::string>>;

Blocks load_cfg(const std::string &cfg_file);

void load_weights(const std::string &weight_file, const Blocks &blocks,
                  std::vector<torch::nn::Sequential> &module_list);

#endif //DARKNET_PARSING_H
