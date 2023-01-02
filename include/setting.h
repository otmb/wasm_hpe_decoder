#pragma once

#include <stdio.h>
const size_t keypointsNumber = 17;
const size_t modelWidth = 192;
const size_t modelHeight = 256;
const size_t heatmap_width = modelWidth / 4;
const size_t heatmap_height = modelHeight / 4;
const float aspect_ratio = modelWidth * 1.0 / modelHeight;
const float pixel_std = 200.0;