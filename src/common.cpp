#include "common.h"

std::vector<float> xywh2cs(float x, float y, float w, float h) {
  std::vector<float> center(2, 0);
  center[0] = x + w * 0.5;
  center[1] = y + h * 0.5;

  if (w > aspect_ratio * h) {
    h = w * 1.0 / aspect_ratio;
  } else if (w < aspect_ratio * h) {
    w = h * aspect_ratio;
  }
  std::vector<float> scale = {static_cast<float>(w * 1.0 / pixel_std), static_cast<float>(h * 1.0 / pixel_std)};
  if (center[0] != -1) {
    std::transform(scale.begin(), scale.end(), scale.begin(),
                   std::bind(std::multiplies<float>(), std::placeholders::_1, 1.25));
  }
  return {center[0], center[1], scale[0], scale[1]};
}

void box2cs(const std::vector<float> & box,
            std::vector<float> & center,
            std::vector<float> & scale){

  float x, y, w, h;
  x = box[0];
  y = box[1];
  w = box[2];
  h = box[3];
  const std::vector<float> & bbox = xywh2cs(x, y, w, h);

  center = { bbox[0], bbox[1] };
  scale = { bbox[2], bbox[3] };

}

float sign(float A){
    return (A>0)-(A<0);
}

std::vector<float> preExecute(cv::Mat image,
                const std::vector<float> & box,
                std::vector<float> & center,
                std::vector<float> & scale)
{
  box2cs(box, center, scale);
  
  std::vector<int> output_size = {modelWidth, modelHeight};
  cv::Mat trans;
  std::vector<float> _scale(scale);
  _scale[0] = scale[0] * 200;
  _scale[1] = scale[1] * 200;
  get_affine_transform(center, _scale, 0, output_size, trans, 0);
  cv::Mat cropped_box;
  cv::warpAffine(image, cropped_box, trans, cv::Size(output_size[0], output_size[1]), cv::INTER_LINEAR);
  
  cv::Mat bgr;
  cv::cvtColor(cropped_box, bgr, cv::COLOR_RGB2BGR);
  
  std::vector<uchar>rawBytes((uchar*)bgr.datastart, (uchar*)bgr.dataend);

  size_t w = modelWidth;
  size_t h = modelHeight;
  std::vector<float> normalizedBuffer( w * h * 3 );

  for (int i = 0; i < w * h; ++i){
    normalizedBuffer[i]             = (float(rawBytes[i * 3 + 0]) / 255.0 - 0.406) / 0.225; // B
    normalizedBuffer[w * h + i]     = (float(rawBytes[i * 3 + 1]) / 255.0 - 0.456) / 0.224; // G
    normalizedBuffer[w * h * 2 + i] = (float(rawBytes[i * 3 + 2]) / 255.0 - 0.485) / 0.229; // R
  }

  return normalizedBuffer;
}

std::vector<float> postExecute(std::vector<float>& heatmap, std::vector<float> & center, std::vector<float> & scale)
{
  std::vector<int> dim = { 1, keypointsNumber, heatmap_height, heatmap_width };
  std::vector<float> coords(keypointsNumber * 2, 0);
  std::vector<float> maxvals(keypointsNumber, 0);
  std::vector<float> preds(keypointsNumber * 3, 0);
  std::vector<int> img_size{heatmap_width, heatmap_height};
  
  get_max_preds(heatmap,
                dim,
                coords,
                maxvals,0,0);
  
  for (int j = 0; j < dim[1]; ++j) {
    int index = j * dim[2] * dim[3];
    int px = int(coords[j * 2] + 0.5);
    int py = int(coords[j * 2 + 1] + 0.5);
    
    if (px > 0 && px < heatmap_width - 1) {
      float diff_x = heatmap[index + py * dim[3] + px + 1] -
                          heatmap[index + py * dim[3] + px - 1];
      coords[j * 2] += sign(diff_x) * 0.25;
    }
    if (py > 0 && py < heatmap_height - 1) {
      float diff_y = heatmap[index + (py + 1) * dim[3] + px] -
                          heatmap[index + (py - 1) * dim[3] + px];
      coords[j * 2+1] += sign(diff_y) * 0.25;
    }
  }
  
  std::vector<float> _scale(scale);
  _scale[0] = scale[0] * 200;
  _scale[1] = scale[1] * 200;
  
  transform_preds(coords, center, _scale, img_size, dim, preds, true);
  
  std::vector<float> results(keypointsNumber * 3, 0);
  
  for (int j = 0; j < dim[1]; ++j) {
    results[j * 3] = preds[j * 3 + 1];
    results[j * 3 + 1] = preds[j * 3 + 2];
    results[j * 3 + 2] = maxvals[j];
  }
  return results;
}
