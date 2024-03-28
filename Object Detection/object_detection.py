# Importing dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import CategoricalCrossentropy
import keras_cv

BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE

print("Dependecies Import Successful.")

###################################################################
# Function for reading Labels and creating dict of {img, cls, bbox}
###################################################################
normalization_layer = tf.keras.layers.Rescaling(1./255)

# a function for converting txt labels file to list
def parse_txt_annot(txt_path):
    file_label = open(txt_path, "r")
    lines = file_label.read().split('\n')
    ids = []
    boxes = []
    classes = []

    # print(int(len(lines)))
    for i in range(0, int(len(lines)-1)):
        objbud = lines[i].split(' ')
        try:
            class_ = int(objbud[1])

            c_x = int(objbud[2])
            c_y = int(objbud[3])
            w1 = int(objbud[4])
            h1 = int(objbud[5])
        except:
            print(objbud, i)
        ids.append(int(objbud[0]))
        boxes.append([c_x, c_y, w1, h1])
        classes.append(class_)

    return np.array(ids), np.array(classes), np.array(boxes)

# a function for creating file paths list
def create_paths_list(path):
    full_path = {}
    images = sorted(os.listdir(path))

    for i in images:
        full_path[int(i.split('.')[0])] = os.path.join(path, i)

    return full_path

# a function for creating a dict format of files
def creating_files(image_dir, labels_dir):

    img_files = create_paths_list(image_dir)
    # annot_files = create_paths_list(annot_files_paths)
    img_id, class_, bbox_ = parse_txt_annot(labels_dir)
    image_paths = []
    bbox = []
    classes = []

    for i in img_files.keys():
        image_paths.append(img_files[i])
        bbox.append(bbox_[np.where(img_id == i)])
        classes.append(class_[np.where(img_id == i)])

    image_paths = tf.ragged.constant(image_paths)
    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)

    return image_paths, classes, bbox

def create_test_data(img_dir):
    img_files = create_paths_list(img_dir)
    #print(img_files)
    image_paths = []
    
    for i in img_files.keys():
        image_paths.append(img_files[i])
        
    image_paths = tf.ragged.constant(image_paths)
    #print(image_paths)
    return image_paths
    
def img_preprocessing_test(img_path):
    #print(img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    #img = tf.image.resize(img, [640, 640])
    return img


############################
# Data Directory Setup
############################
main_dir = '//WAVE/projects/CSEN-342-Wi24/data/pr2/'

dirs            = os.listdir(main_dir)
test_dir_path   = main_dir + dirs[0]
train_dir_path  = main_dir + dirs[1]
val_dir_path    = main_dir + dirs[2]

train_dirs          = os.listdir(train_dir_path)
train_img_dir       = train_dir_path + f'/{train_dirs[1]}'
train_labels_file   = train_dir_path + f'/{train_dirs[0]}'

val_dirs            = os.listdir(val_dir_path)
val_img_dir         = val_dir_path + f'/{val_dirs[0]}'
val_labels_file     = val_dir_path + f'/{val_dirs[1]}'

test_dirs           = os.listdir(test_dir_path)
test_img_dir        = test_dir_path + f'/{test_dirs[0]}'

print("Data Diretories Setup Complete. ")

training = input("do yo want to train the model?[y/n] ")

if training =='y':
    train_img_paths, train_classes, train_bboxes = creating_files(
        train_img_dir, train_labels_file)
    val_img_paths, val_classes, val_bboxes = creating_files(
        val_img_dir, val_labels_file)
    test_img_paths    = create_test_data(test_img_dir)


#####################################################
# Function for Creating Input Data Pipeline
#####################################################  

# reading and resizing images
def img_preprocessing(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    #img = tf.image.resize(img, [512, 512])
    return img

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.8, 1.25),
    bounding_box_format="center_xyWH")

resizing_test = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.8, 1.25))

def load_ds(img_paths, classes, bbox):
    img = img_preprocessing(img_paths)

    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox.to_tensor(default_value=0)}

    return {"images": img, "bounding_boxes": bounding_boxes}

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


# Creating dataset loaders and tf.datasets
if training == 'y':

    train_loader = tf.data.Dataset.from_tensor_slices(
        (train_img_paths, train_classes, train_bboxes))
    train_dataset = (train_loader
                    .map(load_ds, num_parallel_calls=AUTO)
                    .shuffle(BATCH_SIZE*10)
                    .ragged_batch(BATCH_SIZE, drop_remainder=True)
                    .map(resizing, num_parallel_calls=AUTO)
                    .map(dict_to_tuple, num_parallel_calls=AUTO)
                    .prefetch(AUTO))

    val_loader = tf.data.Dataset.from_tensor_slices(
        (val_img_paths, val_classes, val_bboxes))
    val_dataset = (val_loader
                .map(load_ds, num_parallel_calls=AUTO)
                .shuffle(BATCH_SIZE*10)
                .ragged_batch(BATCH_SIZE, drop_remainder=True)
                .map(resizing, num_parallel_calls=AUTO)
                .map(dict_to_tuple, num_parallel_calls=AUTO)
                .prefetch(AUTO))
    
    test_loader  = tf.data.Dataset.from_tensor_slices((test_img_paths))
    test_dataset = (test_loader
                    .map(img_preprocessing_test,num_parallel_calls=AUTO)
                    .ragged_batch(BATCH_SIZE, drop_remainder=True)
                    .map(resizing_test, num_parallel_calls=AUTO)
                    .prefetch(AUTO))
else:
    test_img_paths    = create_test_data(test_img_dir)

    test_loader  = tf.data.Dataset.from_tensor_slices((test_img_paths))
    test_dataset = (test_loader
                    .map(img_preprocessing_test,num_parallel_calls=AUTO)
            .ragged_batch(BATCH_SIZE, drop_remainder=True)
            .map(resizing_test, num_parallel_calls=AUTO)
            .prefetch(AUTO))

class_ids       = ['car', 'medium truck', 'large truck']
class_mapping   = {1: 'car', 2: 'medium truck', 3: 'large truck'}

print("Data Pipeline Ready....")

#################################
# For Data Visualization
#################################
# a function to visualize samples from a dataset

def visualize_dataset(inputs, value_range, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs[0], inputs[1]

    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        rows=2,
        cols=2,
        value_range=value_range,
        y_true=bounding_boxes,
        scale=6,
        font_scale=0.8,
        line_thickness=2,
        dpi=100,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
        true_color=(192, 57, 43))

# examples images and annotations from training daatset
# visualize_dataset(
#     train_dataset, bounding_box_format="center_xyWH", value_range=(0, 255))


############################
# Model Creation
############################
# creating mirrored strategy
stg = tf.distribute.MirroredStrategy()

def freeze(model):
    layers = model.layers
    for i in range(len(layers)-25):
        layers[i].trainable = False
    return model

# creating yolo backbone
with stg.scope():
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco", include_rescaling=True, load_weights=True)

    YOLOV8_model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping),
                                                  bounding_box_format="center_xyWH", backbone=backbone, fpn_depth=2)

    optimizer = Adam(learning_rate=0.01,
                      weight_decay=0.0009)

    #YOLOV8_model = freeze(YOLOV8_model)
    YOLOV8_model.compile(
        optimizer=optimizer, classification_loss='binary_crossentropy', box_loss='ciou')

print("Model Creation Complete.")

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
class SaveBestWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, monitor=['loss', 'val_loss', 'val_class_loss'], mode='min'):
        super(SaveBestWeightsCallback, self).__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best = [np.Inf, np.Inf, np.Inf]

    def on_epoch_end(self, epoch, logs=None):
        current = [logs.get(self.monitor[0]),logs.get(self.monitor[1]), logs.get(self.monitor[2])]
        print(current)
        print(self.best)
        if self.mode == 'min' and current < self.best:
            print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best} to {current}. Saving model weights.")
            self.best = current
            self.model.save_weights(f"{self.save_path}")
        else:
            print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best}.")


if training == 'y':
    load_w = input("Do you want to load the old weights: [y/n]")

    if load_w == 'y':
        YOLOV8_model.load_weights('./P2_call.tf')
        
    print("Enter the number of Epochs: \n")
    epochs = int(input())

    print("Model Training Started.")
    
    hist = YOLOV8_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, 
                            callbacks=[SaveBestWeightsCallback("P2_call.tf"), tensorboard_callback],
                            )
    print("Training Complete.")

    YOLOV8_model.save_weights('./P2.tf', overwrite=False)

    print("Weights Saved.")

    testing = input("Do you want to make predictions using this model:[y/n] ")
    if testing == 'y':
        img_ids     = 0
        img_cls     = []
        img_bbox    = []
        w, h        = 640, 640

        with open('results_trained.txt', 'a') as fp:
            for i in test_dataset.as_numpy_iterator():
                y_pred = YOLOV8_model.predict(i, verbose = 0)
                y_pred = keras_cv.bounding_box.to_ragged(y_pred)

                for img in range(32):
                    print("Processing Image : ", img+img_ids+1)
                    clss = y_pred['classes'][img]

                    if len(clss) > 0:
                        bboxes      = y_pred['boxes'][img]
                        conf_list   = y_pred['confidence'][img]

                        for cls in range(len(clss)):

                            cls_list    = (clss[cls]).numpy()
                            confs       = (conf_list[cls]).numpy()
                            bx_list     = (bboxes[cls]).numpy().tolist()
                            line        = f"{img+img_ids+1} {int(cls_list)} {bx_list[0]/w} {bx_list[1]/h} {bx_list[2]/w} {bx_list[3]/h} {confs}\n"
                            
                            fp.write(line)
                            
                img_ids = img+img_ids+1

######################################################
# Making Predictions on Test Data 
######################################################    
else:

    YOLOV8_model.load_weights('./P2_call.tf')
    img_ids     = 0
    img_cls     = []
    img_bbox    = []
    w, h        = 640, 640

    with open('results.txt', 'a') as fp:
        for i in test_dataset.as_numpy_iterator():
            y_pred = YOLOV8_model.predict(i, verbose = 0)
            y_pred = keras_cv.bounding_box.to_ragged(y_pred)

            for img in range(32):
                print("Processing Image : ", img+img_ids+1)
                clss = y_pred['classes'][img]

                if len(clss) > 0:
                    bboxes      = y_pred['boxes'][img]
                    conf_list   = y_pred['confidence'][img]

                    for cls in range(len(clss)):

                        cls_list    = (clss[cls]).numpy()
                        confs       = (conf_list[cls]).numpy()
                        bx_list     = (bboxes[cls]).numpy().tolist()
                        line        = f"{img+img_ids+1} {int(cls_list)} {bx_list[0]/w} {bx_list[1]/h} {bx_list[2]/w} {bx_list[3]/h} {confs}\n"
                        
                        fp.write(line)
                        
            img_ids = img+img_ids+1
