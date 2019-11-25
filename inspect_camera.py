#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import sys
import time

if __name__ == '__main__':
    # parse options
    parser = argparse.ArgumentParser(description='keras-pi.')
    parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-w', '--weights', default='./model/weights.99.hdf5')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')
    parser.add_argument('-d', '--device', default='normal_cam') # normal_cam /jetson_nano_raspi_cam

    args = parser.parse_args()

    labels = []
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)

    model_pred = model_from_json(open(args.model).read())
    model_pred.load_weights(args.weights)

    # model_pred.summary()
    if args.device == 'normal_cam':
        cam = cv2.VideoCapture(0)
    elif args.device == 'jetson_nano_raspi_cam':
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 \
            ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
            ! videoconvert \
            ! appsink'
        cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
    else:
        print('wrong device')
        sys.exit()

    count_max = 0
    count = 0

    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count > count_max:
            X = []
            img = capture.copy()
            img = cv2.resize(img, (64, 64))
            img = img_to_array(img)
            img = img/255
            X.append(img)
            X = np.asarray(X)
            start = time.time()
            preds = model_pred.predict(X)
            elapsed_time = time.time() - start

            pred_label = ""

            label_num = 0
            tmp_max_pred = 0
            for i in preds[0]:
                if i > tmp_max_pred:
                    pred_label = labels[label_num]
                    tmp_max_pred = i
                label_num += 1

            # Put speed
            speed_info = '%s: %f' % ('speed=', elapsed_time)
            # print(speed_info)
            cv2.putText(capture, speed_info , (10,50), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Put label
            cv2.putText(capture, pred_label, (10,100), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('keras-pi inspector', capture)
            count = 0

    cam.release()
    cv2.destroyAllWindows()
