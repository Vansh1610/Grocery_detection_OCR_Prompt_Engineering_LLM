import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, AUC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import  models



label_list = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
label_to_int = {label: i for i, label in enumerate(label_list)}
int_to_label={i: label for i, label in enumerate(label_list)}

model = None

def load_model():
    global model
    if model is None:
        model = build_model()
        model.load_weights('model.h5')
    return model

def sort_contours(ctrs):
    sorted_ctrs = sorted(ctrs, key=lambda bbox: cv2.boundingRect(bbox)[1])

    sorted_lines = []
    line_group = []
    last_y = cv2.boundingRect(sorted_ctrs[0])[1]

    for contour in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(y - last_y) > 30:
            sorted_lines.append(sorted(line_group, key=lambda bbox: cv2.boundingRect(bbox)[0]))
            line_group = [contour]
        else:
            line_group.append(contour)
        
        last_y = y
    if line_group:
        sorted_lines.append(sorted(line_group, key=lambda bbox: cv2.boundingRect(bbox)[0]))

    return sorted_lines

def test_image(im,threshold_method,min_contour_area,display_boxes,normal_threshold_value):
    output = []
    prev = None
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if threshold_method == "Adaptive":
        im_t = cv2.adaptiveThreshold(im_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    elif threshold_method == "Normal":
        _, im_t = cv2.threshold(im_g, normal_threshold_value, 255, cv2.THRESH_BINARY_INV)
    elif threshold_method == "Otsu":
        _, im_t = cv2.threshold(im_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    ctrs, _ = cv2.findContours(im_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ctrs = [c for c in ctrs if cv2.contourArea(c) > min_contour_area]

    if not ctrs:
        return "No contours found.",im

    model = load_model()
    ctrs = sort_contours(ctrs)



    for line in ctrs:
        line_chr = ""
        for bbox in line:
            x, y, w, h = cv2.boundingRect(bbox)
            if display_boxes:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            im_crop = im_g[y:y+h, x:x+w]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                im_resized = cv2.resize(im_crop, (28, 28))
                im_resized = im_resized.reshape(1, 28, 28, 1)
                im_resized = im_resized / 255.0
                pred = model.predict(im_resized)
                pred_idx = np.argmax(pred, axis=1)[0]
                pred_chr = int_to_label[pred_idx]

                if prev is None:
                    line_chr += pred_chr
                elif (x) - (prev[0] + prev[2]) > 80:
                    line_chr += ' '
                    line_chr += pred_chr
                else:
                    line_chr += pred_chr
                prev = [x, y, w, h]
                cv2.putText(im, pred_chr, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

        output.append(line_chr)
        prev = None 
    return output,im

def build_model():
    model = tf.keras.Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(62, activation='softmax'))

    return model


