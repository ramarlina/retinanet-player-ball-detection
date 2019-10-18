# RetinaNet Player and Ball Detection
This project shows you how you can build a Computer Vision model that can detect players and ball in overhead camera images using RetinaNet. 

This project uses Fizyr's awesome keras-retinanet implementation, and applies it to Haizaha's Soccer Player and Ball Detection free dataset.

## Setup
You will need the Tensorflow and Keras installed in addition to the standard shenanigan.

## The Data
To get the data and the pre-trained weight, run download.sh:

> bash download.sh

This will download everything you need to run the project, including downloading and compiling keras-retinanet

## Inference
To test the pre-trained model on a sample image, just run predict.py
> python predict.py

## Training
To train the model:
> python keras-retinanet/keras_retinanet/bin/train.py --snapshot pre-trained/resnet50_csv_last.h5   csv data/train.csv data/labels.csv --val-annotations data/valid.csv

Before you can use the newly trained model, you'll need to convert it into an inference model:
> python keras-retinanet/keras_retinanet/bin/convert_model.py ./snapshots/resnet50_csv_01.h5  ./snapshots/resnet50_csv_01_inference.h5
