# wasm_hpe_decoder

![result](result.png)

- Model in use
    - Pose Estimation: SimpleBaseline resnet50 192x256(fp16)
    - Object Detection: Yolov7-tiny 600x600(fp16)

[Sample: Video Stream](https://otmb.github.io/wasm_hpe_decoder)

# References

- [microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb)
- [tensorflow/tfjs-models](https://github.com/tensorflow/tfjs-models/blob/master/posenet/src/util.ts)
- [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/lite/src/keypoint_postprocess.cc)
- [openvinotoolkit/open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/human_pose_estimation_demo/cpp/main.cpp)
- [kounoike/webassembly-night-sample](https://github.com/kounoike/webassembly-night-sample/tree/master/run-by-full-wasm)