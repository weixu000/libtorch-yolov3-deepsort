#ifndef DARKNET_H
#define DARKNET_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include "bbox.h"

using namespace std;

Detection write_results(torch::Tensor prediction, float confidence, float nms_conf);

struct Darknet : torch::nn::Module {

public:

    Darknet(const char *conf_file, torch::Device *device);

    map<string, string> *get_net_info();

    void load_weights(const char *weight_file);

    torch::Tensor forward(torch::Tensor x);

private:

    torch::Device *_device;

    vector<map<string, string>> blocks;

    vector<torch::nn::Sequential> module_list;

    // load YOLOv3
    void load_cfg(const char *cfg_file);

    void create_modules();
};

#endif //DARKNET_H