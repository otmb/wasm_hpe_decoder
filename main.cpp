#include <stdio.h>
#include <stdlib.h> 
#include <emscripten.h>
#include "keypoint_postprocess.h"
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>
#include <SDL/SDL.h>
#include "setting.h"
#include "common.h"
#include "human_pose.h"
#include "render_human_pose.h"

SDL_Surface *screen = nullptr;

constexpr int WIDTH = 640;
constexpr int HEIGHT = 480;
int main() {
    printf("run\n");
    SDL_Init(SDL_INIT_VIDEO);
    
    screen = SDL_SetVideoMode(WIDTH, HEIGHT, 32, SDL_SWSURFACE);
    return 0;
}

extern "C" int EMSCRIPTEN_KEEPALIVE preRun(size_t addr, int width, int height, float* center_p, float* scale_p, float* boxes_p){
  
  std::vector<float> box(&boxes_p[0], boxes_p + 4);
  box[2] = box[2] - box[0];
  box[3] = box[3] - box[1];

  auto data = reinterpret_cast<void *>(addr);
  cv::Mat image(height, width, CV_8UC4, data);
  
  std::vector<float> center;
  std::vector<float> scale;
  std::vector<float> normalize = preExecute(image, box, center, scale);

  center_p[0] = center[0];
  center_p[1] = center[1];
  scale_p[0] = scale[0];
  scale_p[1] = scale[1];

  float *norm = normalize.data();
  return (int)norm;
}

extern "C" int EMSCRIPTEN_KEEPALIVE postRun(float* heatmap_p, float* center_p, float* scale_p){
  std::vector<float> center = { center_p[0], center_p[1] };
  std::vector<float> scale = { scale_p[0], scale_p[1] };
  size_t outputSize = heatmap_height * heatmap_width * keypointsNumber;
  std::vector<float> heatmap(&heatmap_p[0], heatmap_p + outputSize);
  std::vector<float> preds = postExecute(heatmap, center, scale);
  float *result = preds.data();
  return (int)result;
}

extern "C" void EMSCRIPTEN_KEEPALIVE poseRender(size_t addr, int width, int height, float* boxes_p, float* keypoints_p, int people_num){

  std::vector<float> keypoints(&keypoints_p[0], keypoints_p + keypointsNumber * people_num * 3);
  std::vector<float> boxes(&boxes_p[0], boxes_p + people_num * 4);

  auto data = reinterpret_cast<void *>(addr);
  cv::Mat image(height, width, CV_8UC4, data);

  std::vector<HumanPose> poses;
  for (int i = 0; i < people_num; ++i) {
    HumanPose pose{
      std::vector<cv::Point2f>(keypointsNumber, cv::Point2f(-1.0f, -1.0f)),
      std::vector<float>(keypointsNumber, 0.0),
      1.0
    };
    for (int j = 0; j < keypointsNumber; ++j) {
      int n = i * keypointsNumber * 3 + j * 3;
      pose.keypoints[j].x = keypoints[n];
      pose.keypoints[j].y = keypoints[n + 1];
      pose.scores[j] = keypoints[n + 2];
    }
    pose.score = std::accumulate(pose.scores.begin(), pose.scores.end(), 0.0) / pose.scores.size();
    poses.push_back(pose);
  }

  cv::Mat bgr;
  cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
  renderHumanPose(poses, bgr);

  for (int j = 0; j < people_num; ++j) {
    std::vector<float> bbox = { boxes[j*4], boxes[j*4+1], boxes[j*4+2], boxes[j*4+3] };
    // cv::rectangle(bgr, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2] + bbox[0], bbox[3] + bbox[1]), cv::Scalar(0,255,0), 2);
    cv::rectangle(bgr, cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]), cv::Scalar(0,255,0), 2);
  }

  cv::cvtColor(bgr, image, cv::COLOR_BGR2RGBA);

  if (SDL_MUSTLOCK(screen))
    SDL_LockSurface(screen);
  cv::Mat dstRGBAImage(height, width, CV_8UC4, screen->pixels);
  image.copyTo(dstRGBAImage);
  if (SDL_MUSTLOCK(screen))
    SDL_UnlockSurface(screen);
  SDL_Flip(screen);
}

