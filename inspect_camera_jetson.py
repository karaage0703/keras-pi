#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from time import sleep

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 \
    ! nvvidconv ! video/x-raw, width=(int)224, height=(int)224, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

if __name__ == '__main__':
    # parse options
    parser = argparse.ArgumentParser(description='keras-pi.')
    parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-w', '--weights', default='./model/weights.99.hdf5')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')

    args = parser.parse_args()

    labels = []
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)

    model_pred = model_from_json(open(args.model).read())
    model_pred.load_weights(args.weights)

    model_pred.summary()

    cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    count = 0

    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        cv2.imshow('keras-jetson inspector', capture)
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count == 1:
            X = []
            img = capture.copy()
            img = cv2.resize(img, (64, 64))
            img = img_to_array(img)
            X.append(img)
            X = np.asarray(X)
            preds = model_pred.predict(X)

            pred_label = ""

            label_num = 0
            for i in preds[0]:
                if i == 1.0:
                    pred_label = labels[label_num]
                    break
                label_num += 1

            print("label=" + pred_label)
            count = 0

    cam.release()
    cv2.destroyAllWindows()
