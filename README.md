# keras-pi
Using Keras for deep learning application

# Dependency
- Anaconda3==4.4.0
- TensorFlow==1.7.0
- Keras==2.1.6

# Setup

# Usage

## Prediction

```sh
$ python pred.py -l ./model/labels.txt -m ./model/mnist_deep_model.json -w ./model/weights.99.hdf5 -t ./data/test.jpg
```

## Prediction on Raspberry Pi

```sh
$ python3 inspect_camera_pi.py -l ./model/labels.txt -m ./model/mnist_deep_model.json -w ./model/weights.99.hdf5 -t ./data/test.jpg
```

# License
This software is released under the MIT License, see LICENSE.

# Authors
karaage0703

# References
