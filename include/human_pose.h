// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

struct HumanPose {
    // HumanPose(const std::vector<cv::Point2f>& keypoints = std::vector<cv::Point2f>(),
    //           const std::vector<float>& scores = std::vector<float>(),
    //           const float& score = 0);

    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;
    float score;
};