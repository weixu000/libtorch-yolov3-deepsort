#ifndef KALMAN_H
#define KALMAN_H

#include "opencv2/video/tracking.hpp"

using StateType = cv::Rect2f;

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker {
public:
    explicit KalmanTracker(StateType initRect);

    StateType predict();

    void update(StateType stateMat);

    StateType get_state() const;

    int time_since_update = 0;
    int id = kf_count++;

private:
    static int kf_count;

    cv::KalmanFilter kf;
    cv::Mat measurement;
};


#endif //KALMAN_H