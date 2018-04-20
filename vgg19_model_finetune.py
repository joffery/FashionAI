import os
import numpy as np
import shutil
from keras.engine import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import VGG16, ResNet50, VGG19
from keras.applications.vgg16 import preprocess_input
from tensorflow import set_random_seed
import time
import sys
import pandas as pd
from utils import *
from skimage import io, transform

labels = ["blouse", "dress", "outwear", "skirt", "trousers"]

label_points = {"blouse": ["neckline_left", "neckline_right", "shoulder_left", "shoulder_right", "center_front",
                           "armpit_left", "armpit_right", "top_hem_left", "top_hem_right", "cuff_left_in",
                           "cuff_left_out", "cuff_right_in", "cuff_right_out"],
                "dress": ["neckline_left", "neckline_right", "shoulder_left", "shoulder_right", "center_front",
                          "armpit_left", "armpit_right", "waistline_left", "waistline_right", "cuff_left_in",
                          "cuff_left_out", "cuff_right_in", "cuff_right_out", "hemline_left", "hemline_right"],
                "outwear": ["neckline_left", "neckline_right", "shoulder_left", "shoulder_right", "armpit_left",
                            "armpit_right", "waistline_left", "waistline_right", "cuff_left_in", "cuff_left_out",
                            "cuff_right_in",
                            "cuff_right_out", "top_hem_left", "top_hem_right"],
                "skirt": ["waistband_left", "waistband_right", "hemline_left", "hemline_right"],
                "trousers": ["waistband_left", "waistband_right", "crotch", "bottom_left_in", "bottom_left_out",
                             "bottom_right_in",
                             "bottom_right_out"]}

train_path = r"fashionAI_key_points_train_20180227/train"
test_path = r"fashionAI_key_points_test_a_20180227/test"
train_img_path = os.path.join(train_path, "Images")
train_anno_path = os.path.join(train_path, "Annotations", "train.csv")
test_img_path = os.path.join(test_path, "Images")
train_anno = pd.read_csv(train_anno_path)
test_anno = pd.read_csv(os.path.join(test_path, "test.csv"))
image_size = (512, 512)

def generate_batch(input_imgs, step, batch_size):
    batch_img = input_imgs[step * batch_size: (step + 1) * batch_size]
    X_train_originsize = []
    X_train = []

    for img_name in batch_img:
        img = io.imread(os.path.join(train_path, img_name[0]))
        X_train_originsize.append(img.shape)
        X_train.append(transform.resize(img, image_size, preserve_range=True))
    # print(X_train_originsize)
    X_train = np.asarray(X_train)
    # print(X_train.shape)
    X_train = preprocess_input(X_train)
    y_train = [[point.split("_")[0:2] for point in img_name[1:]] for img_name in batch_img]
    y_train = np.asarray(y_train, dtype=np.float32)
    for i in range(batch_size):
        y_train[i, :, 0] = 512*y_train[i, :, 0]/X_train_originsize[i][0]
        y_train[i, :, 1] = 512*y_train[i, :, 1]/X_train_originsize[i][1]
    y_train = y_train.reshape((batch_size,-1))
    y_train[y_train>511] = 511
    return X_train, y_train/512

def ResNet50_model():
    pass

def vgg16_model(n_points):
    hidden_dim = 512
    vgg_model = VGG16(include_top=False, input_shape=(512, 512, 3))
    last_layer = vgg_model.get_layer('block5_pool').output
    last_layer = BatchNormalization()(last_layer)
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(n_points, activation=None, name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model

def model_train(model_type, label):
    batch_size = 16
    epochs = 100
    patience = 5
    start = time.time()

    # model prepare
    if model_type == 'vgg16':
        model = vgg16_model(2*len(label_points[label]))

    # freeze the top convolution layers
    for layer in model.layers[ :15]:
        layer.trainable = False

    model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True))

    print(model.summary())

    blouse_points = np.asarray(load_pickle(os.path.join("label/train_pickle", label + ".pickle")))
    blouse_points_num = len(blouse_points)
    blouse_points_train = blouse_points[0:int(0.8*blouse_points_num)]
    blouse_points_valid = blouse_points[int(0.8*blouse_points_num):]
    max_iter = int(len(blouse_points_train) / batch_size - 1)
    max_valid_iter = int(len(blouse_points_valid) / batch_size - 1)

    train_losses = []
    valid_losses = []

    for e in range(1, epochs+1):
        start = time.time()
        print('epoch', e)
        blouse_points_train = train_shuffle(blouse_points_train)
        blouse_points_valid = train_shuffle(blouse_points_valid)

        for step in range(max_iter):
            X_train, y_train = generate_batch(blouse_points_train, step, batch_size)
            train_loss = model.train_on_batch(X_train, y_train)
            if (step % 10 == 0):
                valid_step = int(step/10)
                X_valid, y_valid = generate_batch(blouse_points_valid, valid_step, batch_size)
                valid_loss = model.evaluate(X_valid, y_valid)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                print("train_loss", train_loss)
                print("valid_loss", valid_loss)

    # save model
    if not os.path.exists('model0408'):
        os.mkdir('model0408')
    dump_pickle(os.path.join('model0408', model_type+"_"+label+'_loss.pickle'), [train_losses, valid_losses])
    model.save(os.path.join('model0408', model_type+"_"+label+'.h5'))
    del model

if __name__ == '__main__':
    labels = ["blouse", "dress", "outwear", "skirt", "trousers"]
    gpu_num = sys.argv[1]
    print("gpu_num:",gpu_num)
    model_train(model_type='vgg16', label = labels[int(gpu_num)])
    if gpu_num == str(0):
        model_train(model_type='vgg16', label=labels[-1])