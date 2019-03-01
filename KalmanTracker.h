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

    int m_time_since_update = 0;
    int m_hits = 0;
    int m_hit_streak = 0;
    int m_age = 0;
    int m_id = kf_count++;

private:
    static int kf_count;

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType> m_history;
};


#endif //KALMAN_H