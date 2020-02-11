# TensorFlow YOLOV3
# Python Package "TFYOLOv3"

### Pure TensorFlow 2.0 implementation of the YOLOv3 Object-Detection network and its variations
#### Features
- CPU and GPU use
- Training, inference, and evaluation usage
- Distillation experiements
- Tensorflow 2.0 Best Practices
- Proper packaging

#### Installation Instructions
##### *tensorflow-gpu* v2.0.0
at this time, tensorflow-gpu v2.0.0 requires CUDA 10.0 and cuDNN 7.6.4
For arch users, follow the following links and download the proper packages. (
- [cuda](https://archive.org/download/archlinux_pkg_cuda)
  - [pk.tar.xz link](https://archive.org/download/archlinux_pkg_cuda/cuda-10.0.130-2-x86_64.pkg.tar.xz)
- [cudnn] (https://archive.org/download/archlinux_pkg_cudnn)
  - [pkg.tar.xz link](https://archive.org/download/archlinux_pkg_cudnn/cudnn-7.6.4.38-1-x86_64.pkg.tar.xz)
 Running `sudo pacman -U pkg-file-name`

 For ubunutu users, follow the following [link](https://www.tensorflow.org/install/gpu)

 **I believe CUDA 10.1 is now supported however I have not tried this**

##### package installation - from source
- git clone this repository
- pip install the `requirements.txt`
- pip install the package

#### Usage
- TODO

#### TODO
- [ ] Build the network - a few issues outstanding
  - [x] Darknet53
  - [x] YOLOv3
    - Need to build output layers for training/testing (non-trivial)
  - [ ] YOLOv3-tiny + Darknet backbone
- [ ] YOLOv3 pre-trained weights loading
- [ ] YOLOv3-tiny pre-trained weights loading
- [x] GPU Acceleration
- [ ] Inference demo code
- [ ] Training pipeline
  - [ ] eager mode (`tf.GradientTape`)
  - [ ] graph mode (`model.fit`)
- [ ] Distillation module
- [x] Tensorflow Best Practices
  - subclass `tf.keras.Model` and `tf.keras.layers.Layer`
