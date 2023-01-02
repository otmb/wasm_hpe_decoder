#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "keypoint_postprocess.h"
#include "setting.h"

std::vector<float> xywh2cs(float x, float y, float w, float h);
void box2cs(const std::vector<float> & box,
            std::vector<float> & center,
            std::vector<float> & scale);
std::vector<float> preExecute(cv::Mat image,
                const std::vector<float> & box,
                std::vector<float> & center,
                std::vector<float> & scale);
std::vector<float> postExecute(std::vector<float>& heatmap, std::vector<float> & center, std::vector<float> & scale);