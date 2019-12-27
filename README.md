# TensorFlow YOLOV3

### Pure TensorFlow 2.0 implementation of the YOLOv3 Object-Detection network and its variations
#### Features
- CPU and GPU use
- Training, inference, and evaluation usage
- Distillation experiements
- Proper packaging which allows use of inference in a system of your own design

#### TODO
- [ ] Build the network
  - [x] Darknet53
  - [ ] YOLOv3
    - Need to build output layers for training/testing (non-trivial)
  - [ ] YOLOv3-tiny + Darknet bacbone
- [ ] YOLOv3 pre-trained weights loading
- [ ] YOLOv3-tiny pre-trained weights loading
- [ ] GPU Acceleration
- [ ] Inference demo code
- [ ] Training pipeline
  - [ ] eager mode (tf.GradientTape)
  - [ ] graph mode (model.fit)
- [ ] Distillation module
