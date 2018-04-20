import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
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

def construct_pickle():
    if not os.path.exists("train_pickle"):
        os.mkdir("train_pickle")
    if not os.path.exists("test_pickle"):
        os.mkdir("test_pickle")

    for lab in labels:
        blouse = train_anno[train_anno["image_category"]==lab]
        blouse_points = blouse[["image_id"]+label_points[lab]] .values
        blouse_points_clean = []
        for i in range(len(blouse_points)):
            if '-1_-1_-1' in blouse_points[i]:
                continue
            blouse_points_clean.append(blouse_points[i])
        dump_pickle(os.path.join("train_pickle", lab+".pickle"), blouse_points_clean)

    for lab in labels:
        blouse = test_anno[test_anno["image_category"]==lab]
        blouse_points = blouse[["image_id"]] .values
        dump_pickle(os.path.join("test_pickle", lab+".pickle"), blouse_points)

def img_visualization():
    for lab in labels:
        img_list = load_pickle(os.path.join("train_pickle", lab+".pickle"))
        img_vis(phase="train", img_list=img_list, label=lab)

# can't load all images (memory overflow)
# def construct_train_test():
#     if not os.path.exists("image/train_pickle"):
#         os.mkdir("image/train_pickle")
#     if not os.path.exists("image/test_pickle"):
#         os.mkdir("image/test_pickle")
#
#     for lab in labels:
#         imgs = []
#         img_list = os.listdir(os.path.join(train_img_path, lab))
#         for j, img_name in enumerate(img_list):
#             if j%10 == 0:
#                 print(j)
#             img = io.imread(os.path.join(train_img_path, lab, img_name))
#             img = transform.resize(img, (512,512), preserve_range=True)
#             imgs.append(img)
#         dump_pickle(os.path.join("image/train_pickle", lab + ".pickle"), imgs)

if __name__ == '__main__':
    # construct_pickle()
    # img_visualization()
    # blouse_points = load_pickle(os.path.join("label/train_pickle", "blouse.pickle"))
    # print(blouse_points)
    img = io.imread("test.jpg")
    print(img.shape)