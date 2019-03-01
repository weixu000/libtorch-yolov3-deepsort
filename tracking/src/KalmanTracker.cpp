#include "KalmanTracker.h"

using namespace cv;

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
static inline StateType get_rect_xysr(const Mat &xysr) {
    auto cx = xysr.at<float>(0, 0), cy = xysr.at<float>(1, 0), s = xysr.at<float>(2, 0), r = xysr.at<float>(3, 0);
    float w = sqrt(s * r);
    float h = s / w;
    float x = (cx - w / 2);
    float y = (cy - h / 2);

    if (x < 0 && cx > 0)
        x = 0;
    if (y < 0 && cy > 0)
        y = 0;

    return StateType(x, y, w, h);
}

int KalmanTracker::kf_count = 0;

KalmanTracker::KalmanTracker(StateType initRect) {
    int stateNum = 7;
    int measureNum = 4;
    kf = KalmanFilter(stateNum, measureNum, 0);

    measurement = Mat::zeros(measureNum, 1, CV_32F);

    kf.transitionMatrix = (Mat_<float>(stateNum, stateNum)
            <<
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.statePost.at<float>(0, 0) = initRect.x + initRect.width / 2;
    kf.statePost.at<float>(1, 0) = initRect.y + initRect.height / 2;
    kf.statePost.at<float>(2, 0) = initRect.area();
    kf.statePost.at<float>(3, 0) = initRect.width / initRect.height;
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict() {
    if (time_since_update > 0)
        hit_streak = 0;
    ++time_since_update;

    StateType predictBox = get_rect_xysr(kf.predict());

    return predictBox;
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat) {
    time_since_update = 0;
    ++hit_streak;

    // measurement
    measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
    measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
    measurement.at<float>(2, 0) = stateMat.area();
    measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

    // update
    kf.correct(measurement);
}

// Return the current state vector
StateType KalmanTracker::get_state() const {
    return get_rect_xysr(kf.statePost);
}
