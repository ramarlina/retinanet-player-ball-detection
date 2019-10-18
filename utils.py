# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

def load_model(model_path): 
   return models.load_model(model_path, backbone_name='resnet50')

def view_detections(image_path, csv_file):
    filename = os.path.basename(image_path)
    image = read_image_bgr(image_path)
    
    annotations = [a for a in [i.split(",") for i in open(csv_file).read().split("\n")] if filename in a[0]]
    
    boxes = np.vstack([i[1:-1] for i in annotations]).astype("i")
    
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    for box in boxes: 

        color = label_color(1)

        b = box.astype(int)
        draw_box(draw, b, color=color)
        
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.savefig("ouptut_view_detections.png")

def run_detection(model, filepath, labels_file="data/labels.csv"):
  
    labels = [i.split(",") for i in open(labels_file).read().split("\n")]
    labels_to_names = dict([(int(str_id), name) for name, str_id in labels])
    
    image = read_image_bgr(filepath) 

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        print(box, score, label)
        if (label == 0 and score < .3) or score < .5: 
            break
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.savefig("ouptut_run_detection.png")