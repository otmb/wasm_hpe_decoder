#include "render_human_pose.h"

// https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/multi_channel_human_pose_estimation_demo/cpp/render_human_pose.cpp

void renderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image)
{

  CV_Assert(image.type() == CV_8UC3);

  static const cv::Scalar colors[keypointsNumber] =
  {
    cv::Scalar(255, 0, 0),
    cv::Scalar(255, 85, 0),
    cv::Scalar(255, 170, 0),
    cv::Scalar(255, 255, 0),
    cv::Scalar(170, 255, 0),
    cv::Scalar(85, 255, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 85),
    cv::Scalar(0, 255, 170),
    cv::Scalar(0, 255, 255),
    cv::Scalar(0, 170, 255),
    cv::Scalar(0, 85, 255),
    cv::Scalar(0, 0, 255),
    cv::Scalar(85, 0, 255),
    cv::Scalar(170, 0, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(255, 0, 170),
  };
  /*
   0: nose        1: l eye      2: r eye    3: l ear   4: r ear
   5: l shoulder  6: r shoulder 7: l elbow  8: r elbow
   9: l wrist    10: r wrist    11: l hip   12: r hip  13: l knee
   14: r knee    15: l ankle    16: r ankle
   */
  static const std::pair<int, int> keypointsOP[] = {
    {0, 1}, // nose , l_eye
    {0, 2}, // nose , r_eye
    {1, 3},
    {2, 4},
    {2, 4},
    {5, 7}, // l shoulder l elbow
    {7, 9}, // l elbow l wrist
    {6, 8}, // r shoulder r elbow
    {8, 10},// r elbow r wrist
    {11, 13},
    {13, 15},
    {12, 14},
    {14, 16},
    {5, 6}, // l shoulder r shoulder
    {11, 12}, //
    {5, 11},
    {6, 12},
  };
  
  const int stickWidth = 2;
  const cv::Point2f absentKeypoint(-1.0f, -1.0f);
  for (auto& pose : poses) {
    for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++) {
      if (pose.keypoints[keypointIdx] != absentKeypoint) {
        cv::circle(image, pose.keypoints[keypointIdx], 1, colors[keypointIdx], -1);
      }
    }
  }
  
  std::vector<std::pair<int, int>> limbKeypointsIds;
  if (!poses.empty()) {
    limbKeypointsIds.insert(limbKeypointsIds.begin(), std::begin(keypointsOP), std::end(keypointsOP));
  }
  
  cv::Mat pane = image.clone();
  for (auto pose : poses) {
    for (const auto& limbKeypointsId : limbKeypointsIds) {
      std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                                                        pose.keypoints[limbKeypointsId.second]);
      if (limbKeypoints.first == absentKeypoint || limbKeypoints.second == absentKeypoint) {
        continue;
      }
      
      float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
      float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
      cv::Point difference = limbKeypoints.first - limbKeypoints.second;
      double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
      int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
      std::vector<cv::Point> polygon;
      cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth), angle, 0, 360, 1, polygon);
      cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
    }
  }
  cv::addWeighted(image, 0.4, pane, 0.6, 0, image);
}