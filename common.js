async function loadImage(src) {
  const imgBlob = await fetch(src).then((resp) => resp.blob());
  const img = await createImageBitmap(imgBlob);
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height);
}

async function pose(poseModel, image, boxes, modelHeight = 256, modelWidth = 192, keypointsNumber = 17, inputName="", outputName=""){
  const centerPointer = createFloatHeap(2);
  const scalePointer = createFloatHeap(2);
  
  // const box = [269.44891357421875, 124.16687774658203, 514.134765625, 386.0028076171875];
  // const boxes = [269.44891357421875, 124.16687774658203, 514.134765625, 386.0028076171875, 
  //               45.96035385131836, 284.569580078125, 156.86953735351562, 389.1776123046875,
  //               181.7513885498047, 290.7883605957031, 252.79049682617188, 384.8514404296875, 
  //               312.6956481933594, 273.7389831542969, 411.21160888671875, 392.89984130859375, 
  //               473.4137878417969, 314.5270080566406, 573.98828125, 395.65576171875]
  let keypointBuffer = [];
  const boxesPointer = Module._malloc(boxes.length * 4);
  Module.HEAPF32.set(new Float32Array(boxes), boxesPointer / 4);

  const modelInputSize = modelWidth * modelHeight * 3;
  const peopleNum = boxes.length / 4;
  
  const imagePointer = Module._malloc(image.data.length);
  Module.HEAPU8.set(image.data, imagePointer);

  for (let i = 0; i < peopleNum; i++) {
    const box = [boxes[i*4], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]];
    const boxPointer = Module._malloc(4 * 4);
    Module.HEAPF32.set(new Float32Array(box), boxPointer / 4);

    const inputPointer = _preRun(imagePointer, image.width, image.height, centerPointer, scalePointer, boxPointer);
    const inputData = new Float32Array(Module.HEAPF32.buffer, inputPointer, modelInputSize);
    const x = tf.tidy(() => {
      return tf.tensor1d(inputData).reshape([1, 3, modelHeight, modelWidth]);
    });
    const result = (inputName == "") ? poseModel.predict(x) : await poseModel.executeAsync({[inputName]: x }, outputName);
    const arr = await result.data();
    x.dispose();
    result.dispose();
    
    const heatmapPointer = Module._malloc(arr.length * 4);
    Module.HEAPF32.set(arr, heatmapPointer / 4);

    const keypointPointer = _postRun(heatmapPointer, centerPointer, scalePointer);
    const keypointData = new Float32Array(Module.HEAPF32.buffer, keypointPointer, keypointsNumber * 3);
    keypointBuffer = keypointBuffer.concat(Array.from(keypointData));

    _free(boxPointer);
    _free(heatmapPointer);
  }
  // Render
  const keypointBufferPointer = Module._malloc(keypointBuffer.length * 4);
  Module.HEAPF32.set(new Float32Array(keypointBuffer), keypointBufferPointer / 4);
  _poseRender(imagePointer, image.width, image.height, boxesPointer, keypointBufferPointer, peopleNum);

  _free(centerPointer);
  _free(scalePointer);
  _free(imagePointer);
  _free(boxesPointer);
  _free(keypointBufferPointer);
}

async function detection(detModel, image, modelHeight, modelWidth){
  let boxes = [];
  const {resized, ratio, padding} = padAndResizeTo(image, modelHeight, modelWidth);
  const { x, mean } = tf.tidy(() => {
    const x = resized.div(255).transpose([2,0,1]).reshape([1, 3, modelHeight, modelWidth]);
    const mean = x.mean().dataSync()[0];
    return { x, mean };
  });
  
  if (mean < 0.1){
    resized.dispose();
    x.dispose();
    return;
  }
  const feedDict = {"images": x };
  const preds = await detModel.executeAsync(feedDict, "output");
  const data = await preds.data();
  resized.dispose();
  x.dispose();
  preds.dispose();
  for (let i=0; i<data.length / 7; i++){
    const [ batch_id,x0,y0,x1,y1,cls_id,score ] = data.slice(i*7, i*7+7);
    if (cls_id == 0){
      let box = [
        x0 * ratio - padding.left,
        y0 * ratio - padding.bottom,
        x1 * ratio - padding.left,
        y1 * ratio - padding.bottom,
      ];
      boxes = boxes.concat(Array.from(box));
    }
  }
  // const box = [269.44891357421875, 124.16687774658203, 514.134765625, 386.0028076171875];
  // const boxes = [269.44891357421875, 124.16687774658203, 514.134765625, 386.0028076171875, 
  //               45.96035385131836, 284.569580078125, 156.86953735351562, 389.1776123046875,
  //               181.7513885498047, 290.7883605957031, 252.79049682617188, 384.8514404296875, 
  //               312.6956481933594, 273.7389831542969, 411.21160888671875, 392.89984130859375, 
  //               473.4137878417969, 314.5270080566406, 573.98828125, 395.65576171875]
  
  return boxes;
}

function padAndResizeTo(input, targetH, targetW) {

  const [height, width] = getInputTensorDimensions(input);
  const targetAspect = targetW / targetH;
  const aspect = width / height;
  let [padT, padB, padL, padR] = [0, 0, 0, 0];
  if (aspect < targetAspect) {
    // pads the width
    padT = 0;
    padB = 0;
    padL = Math.round(0.5 * (targetAspect * height - width));
    padR = Math.round(0.5 * (targetAspect * height - width));
  } else {
    // pads the height
    padT = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padB = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padL = 0;
    padR = 0;
  }
  const ratio = Math.max(height/ targetH, width / targetW);

  const resized = tf.tidy(() => {
    let imageTensor = imageToTensor(input);
    imageTensor = tf.pad3d(imageTensor, [[padT, padB], [padL, padR], [0, 0]]);

    return tf.image.resizeBilinear(imageTensor, [targetH, targetW]);
  });

  return {resized, ratio, padding: {top: padT, left: padL, right: padR, bottom: padB}};
}

function createFloatHeap(size){
  const offset = Module._malloc(size * 4);
  Module.HEAPF32.set(new Float32Array(size), offset / 4);
  return offset;
}

function imageToTensor(img) {
  return img instanceof tf.Tensor ? img : tf.browser.fromPixels(img);
}

function getInputTensorDimensions(input){
  return input instanceof tf.Tensor ? 
      [input.shape[0], input.shape[1]] : [input.height, input.width];
}