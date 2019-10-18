mkdir data
mkdir pre-trained
 
wget https://playground.haizaha.com/soccer-player-and-ball-localization/retinanet_csv.zip -O data/retinanet_csv.zip 
cd data
unzip retinanet_csv.zip  
cd ..

wget https://playground.haizaha.com/soccer-player-and-ball-localization/resnet50_csv_last.h5 -O pre-trained/resnet50_csv_last.h5

git clone https://github.com/fizyr/keras-retinanet.git
cd keras-retinanet 
pip3 install .
python3 setup.py build_ext --inplace --user
cd ..