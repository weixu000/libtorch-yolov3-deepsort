# libtorch-yolov3
A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++. It's fast, easy to be integrated to your production, and supports CPU and GPU computation. Enjoy ~

The code is based on the [pytorch version](https://github.com/ayooshkathuria/pytorch-yolo-v3), I rewritten it with C++.

# SORT
The former code is from [walktree](https://github.com/walktree/libtorch-yolov3).
However, I also merged [C++ version of SORT](https://github.com/mcximing/sort-cpp) to do tracking.

A similar software in Python is [here](https://github.com/weixu000/pytorch-yolov3), which also rewrite form [the most starred version](https://github.com/ayooshkathuria/pytorch-yolo-v3) and [SORT](https://github.com/abewley/sort)

# Performance
Currently on a GTX 1060 6G it consumes about 1G RAM and have 37 FPS.

The video I test is [TownCentreXVID.avi](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi).

# GUI
With [Dear ImGui](https://github.com/ocornut/imgui), I developed the GUI module for visualization of results.

# TODO
Still under construction...