#ifndef DARKNET_H
#define DARKNET_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

#include "Detector.h"

struct Detector::Darknet : torch::nn::Module {
public:
    explicit Darknet(const std::string &cfg_file);

    const std::map<std::string, std::string> &net_info() {
        assert(!blocks.empty() && blocks[0]["type"] == "net");
        return blocks[0];
    }

    void load_weights(const std::string &weight_file);

    torch::Tensor forward(torch::Tensor x);

private:
    std::vector<std::map<std::string, std::string>> blocks;

    std::vector<torch::nn::Sequential> module_list;

    void create_modules();
};

#endif //DARKNET_H