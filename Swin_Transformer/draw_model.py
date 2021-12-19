# -*- coding: utf-8 -*-
"""

@project: Swin_Transformer
@author: xu yushen
@create_time: 2021-12-16 23:46:48
@file: draw_model.py
"""
import tensorflow as tf
import os
import json
import glob
import numpy as np
from keras.utils.vis_utils import plot_model
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from model import swin_tiny_patch4_window7_224 as create_model
model = create_model(num_classes=num_classes)
model.build([1, 224, 224, 3])

weights_path = './save_weights/model.ckpt'
assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
model.load_weights(weights_path)

os.environ["PATH"] += os.pathsep + r'D:\Program Files (x86)\Graphviz\bin'
plot_model(model, to_file='./Model/UNet.png', show_shapes=True)