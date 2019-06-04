# keras-pi
Using Keras for deep learning application

# Dependency
## Mac
- Anaconda3==4.4.0
- TensorFlow==1.13.1
- Open CV

## Raspberry Pi
- TensorFlow==1.13.1
- Open CV

## Jetson Nano
- TensorFlow==1.13.1
- Open CV

# Setup

# Usage

## Prediction

```sh
$ python pred.py -l='./model/labels.txt' -m='./model/mnist_deep_model.json' -w='./model/weights.99.hdf5' -t='./data/test.jpg'
```

## Prediction on Mac
```sh
$ python inspect_camera.py -l='./model/labels.txt' -m='./model/mnist_deep_model.json' -w='./model/weights.99.hdf5'
```

## Prediction on Raspberry Pi with raspi cam

```sh
$ python3 inspect_camera_pi.py -l='./model/labels.txt' -m='./model/mnist_deep_model.json' -w='./model/weights.99.hdf5'
```

## Prediction on Jetson Nano with raspi cam
```sh
$ python3 inspect_camera.py -l='./model/labels.txt' -m='./model/mnist_deep_model.json' -w='./model/weights.99.hdf5' -d='jetson_nano_raspi_cam'
```

# License
This software is released under the MIT License, see LICENSE.

# Authors
karaage0703

# References
