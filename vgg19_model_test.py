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
import matplotlib.pyplot as plt

labels = ["blouse", "dress", "outwear", "skirt", "trousers"]

def loss_vis(model_dir, label):
    train_loss, valid_loss = load_pickle(os.path.join(model_dir, "vgg16_"+label+"_loss.pickle"))

    num_steps = len(train_loss)
    plt.figure()
    plt.plot(range(num_steps), train_loss, "r")
    plt.plot(range(num_steps), valid_loss, "b")
    plt.xlabel("steps")
    plt.ylabel("losses")
    plt.legend(["train_loss", "valid_loss"])
    plt.show()
    plt.clf()

if __name__ == '__main__':
    model_dir = "model0408"
    label = "outwear"
    loss_vis(model_dir, label)




