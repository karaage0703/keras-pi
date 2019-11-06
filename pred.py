#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import model_from_json

if __name__ == "__main__":
    # parse options
    parser = argparse.ArgumentParser(description='keras-pi.')
    parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-w', '--weights', default='./model/weights.99.hdf5')
    parser.add_argument('-t', '--testfile', default='./data/test.jpg')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')

    args = parser.parse_args()

    labels = []
    # with open(backup_dir + '/labels.txt','r') as f:
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)

    model_pred = model_from_json(open(args.model).read())
    model_pred.load_weights(args.weights)

    # model_pred.summary()

    X = []
    img_path = args.testfile
    img = img_to_array(load_img(img_path, target_size=(64,64)))
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
