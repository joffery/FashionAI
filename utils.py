import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
from skimage import io, transform
import random
from random import randint
from PIL import Image, ImageEnhance

def img_vis(phase, img_list, label):
    if phase == 'train':
        dir_path = r"fashionAI_key_points_train_20180227\train"
    else:
        dir_path = r"fashionAI_key_points_test_a_20180227\test"

    fig = plt.figure()

    for i in range(len(img_list)):
        img = img_list[i]
        sample = {}
        sample['image'] = io.imread(os.path.join(dir_path, img[0]))
        lms = []
        points_num = len(img)-1
        for pn in range(points_num):
            lms.append([int(img[pn+1].split("_")[0]), int(img[pn+1].split("_")[1])])
        lms = np.asarray(lms)
        sample["landmarks"] = lms
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break

def dump_pickle(pickle_name, lists):
    with open(pickle_name, 'wb') as f:
        pickle.dump(lists, f, protocol=-1)

def load_pickle(pickle_name):
    with open(pickle_name, 'rb') as f:
        return(pickle.load(f, encoding="bytes"))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(1)  # pause a bit so that plots are updated


def random_crop(img, width, height):
    width1 = randint(0, img.size[0] - width)
    height1 = randint(0, img.size[1] - height)
    width2 = width1 + width
    height2 = height1 + height
    img = img.crop((width1, height1, width2, height2))
    return img

def random_flip_left_right(img):
    prob = randint(0, 1)
    if prob == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_contrast(img, lower=0.2, upper=1.8):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(factor)
    return img


def random_brightness(img, lower=0.6, upper=1.4):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Brightness(img)
    img = img.enhance(factor)
    return img

def random_color(img, lower=0.6, upper=1.5):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Color(img)
    img = img.enhance(factor)
    return img

#itertools function using mini-batch, 先不用这个了
def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt

def train_shuffle(train_imgs):
    perm = np.arange(len(train_imgs))
    np.random.seed(10)
    np.random.shuffle(perm)
    return  train_imgs[perm]

