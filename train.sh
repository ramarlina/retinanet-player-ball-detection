# training the model
python3 keras-retinanet/keras_retinanet/bin/train.py --steps 10 --epochs 1 --gpu 0 --snapshot pre-trained/resnet50_csv_last.h5   csv data/train.csv data/labels.csv --val-annotations data/valid.csv

# converting to inference model
python3 keras-retinanet/keras_retinanet/bin/convert_model.py ./snapshots/resnet50_csv_01.h5  ./snapshots/resnet50_csv_01_inference.h5