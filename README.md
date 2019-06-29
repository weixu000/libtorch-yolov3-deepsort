# Overview
*It is for my undergrad thesis in Tsinghua University.*

There are four modules in the project:

- Detection: YOLOv3
- Tracking: SORT and DeepSORT
- Processing: Run detection and tracking, then display and save the results (a compressed video, a few snapshots for each target)
- GUI: Display the results

# YOLOv3
A Libtorch implementation of the YOLO v3 object detection algorithm, written with modern C++.

The code is based on the [walktree](https://github.com/walktree/libtorch-yolov3).

The config file in .\models can be found at [Darknet](https://github.com/pjreddie/darknet/tree/master/cfg).

# SORT
I also merged [SORT](https://github.com/mcximing/sort-cpp) to do tracking.

A similar software in Python is [here](https://github.com/weixu000/pytorch-yolov3), which also rewrite form [the most starred version](https://github.com/ayooshkathuria/pytorch-yolo-v3) and [SORT](https://github.com/abewley/sort)

## DeepSORT
Recently I reimplement [DeepSORT](https://github.com/nwojke/deep_sort) which employs another CNN for re-id.
It seems it gives better result but also slows the program a bit.
Also, a PyTorch version is available at [ZQPei](https://github.com/ZQPei/deep_sort_pytorch), thanks!

# Performance
Currently on a GTX 1060 6G it consumes about 1G RAM and have 37 FPS.

The video I test is [TownCentreXVID.avi](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi).

# GUI
With [wxWidgets](https://www.wxwidgets.org/), I developed the GUI module for visualization of results.

Previously I used [Dear ImGui](https://github.com/ocornut/imgui).
However, I do not think it suits my purpose.

# Pre-trained network
This project uses pre-trained network weights from others
- [YOLOv3](https://pjreddie.com/media/files/yolov3.weights)
- [YOLOv3-tiny](https://pjreddie.com/media/files/yolov3-tiny.weights)
- [DeepSORT](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)

# How to build
This project requires [LibTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), [wxWidgets](https://www.wxwidgets.org/) and [CMake](https://cmake.org/) to build.

LibTorch can be easily integrated with CMake, but there are a lot of strange things...

On Ubuntu 16.04, I use `apt install` to install the others. Everything is fine.
On Windows 10 + Visual Studio 2017, I use the latest stable version of the others from their official websites.
