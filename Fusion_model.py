# -*- coding: utf-8 -*-
"""

@project: Respiratory_disease_detection_Using_Transformer
@author: xu yushen
@create_time: 2021-12-19 13:12:56
@file: Fusion_model.py
"""
import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from ViT.vit_model import vit_base_patch16_224_in21k as create_model


def main():
    num_classes = 8
    im_height = im_width = 224

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # resize image
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # read image
    img = np.array(img).astype(np.float32)

    # preprocess
    img = (img / 255. - 0.5) / 0.5

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    Vit_model = create_model(num_classes=num_classes, has_logits=False)
    Vit_model.build([1, 224, 224, 3])

    weights_path = './save_weights/model.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    Vit_model.load_weights(weights_path)

    ViT_result = np.squeeze(Vit_model.predict(img, batch_size=1))
    ViT_result = tf.keras.layers.Softmax()(ViT_result)

    # img_path = "./jpg/COPD/COPD11.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # #img = tf.io.read_file(img_path)
    # # resize image
    # img = img.resize((im_width, im_height))
    #
    # plt.imshow(img)
    #
    # # read image
    # img = np.array(img).astype(np.float32)
    #
    # # preprocess
    # print(img)
    # img = (img / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, im_width, im_height)
    # img = tf.image.random_flip_left_right(img)
    img = (img / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    Swin_model = create_model(num_classes=num_classes)
    Swin_model.build([1, im_height, im_width, 3])

    weights_path = './save_weights/model.ckpt'
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    Swin_model.load_weights(weights_path)

    Swin_result = np.squeeze(Swin_model.predict(img, batch_size=1))
    Swin_result = tf.keras.layers.Softmax()(Swin_result)

    # 模型融合
    result = (ViT_result+Swin_result)/2

    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
