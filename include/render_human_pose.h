#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "human_pose.h"
#include "setting.h"

void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image);