#include "KalmanTracker.h"

using namespace cv;

namespace {
    // Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
    cv::Rect2f get_rect_xysr(const Mat &xysr) {
        auto cx = xysr.at<float>(0, 0), cy = xysr.at<float>(1, 0), s = xysr.at<float>(2, 0), r = xysr.at<float>(3, 0);
        float w = sqrt(s * r);
        float h = s / w;
        float x = (cx - w / 2);
        float y = (cy - h / 2);

        return cv::Rect2f(x, y, w, h);
    }
}

int KalmanTracker::count = 0;

KalmanTracker::KalmanTracker() {
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
}

void KalmanTracker::init(cv::Rect2f initRect) {
    // initialize state vector with bounding box in [cx,cy,s,r] style
    kf.statePost.at<float>(0, 0) = initRect.x + initRect.width / 2;
    kf.statePost.at<float>(1, 0) = initRect.y + initRect.height / 2;
    kf.statePost.at<float>(2, 0) = initRect.area();
    kf.statePost.at<float>(3, 0) = initRect.width / initRect.height;
}

// Predict the estimated bounding box.
void KalmanTracker::predict() {
    ++time_since_update;

    kf.predict();
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(cv::Rect2f stateMat) {
    time_since_update = 0;
    ++hits;

    if (_state == TrackState::Tentative && hits > n_init) {
        _state = TrackState::Confirmed;
        _id = count++;
    }

    // measurement
    measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
    measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
    measurement.at<float>(2, 0) = stateMat.area();
    measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

    // update
    kf.correct(measurement);
}

void KalmanTracker::miss() {
    if (_state == TrackState::Tentative) {
        _state = TrackState::Deleted;
    } else if (time_since_update > max_age) {
        _state = TrackState::Deleted;
    }
}

// Return the current state vector
cv::Rect2f KalmanTracker::rect() const {
    return get_rect_xysr(kf.statePost);
}
