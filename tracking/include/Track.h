#ifndef DEFINES_H
#define DEFINES_H

#include <opencv2/opencv.hpp>

struct Track {
    int id;
    cv::Rect2f box;
};


#endif //DEFINES_H
