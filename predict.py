from utils import load_model, run_detection

def main():
  model = load_model('./snapshots/resnet50_csv_01_inference.h5')
  run_detection(model, "data/images/frame_0001.jpg", "data/labels.csv")

if __name__=='__main__':
    main()