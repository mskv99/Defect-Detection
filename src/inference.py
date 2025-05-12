from ultralytics import YOLO
import fire
import torch

def main(inf_dir: str = None,
         weights: str = None,
         imgsz: int = 640,
         conf: int = 0.25,
         save: bool = True,
         project: str = '/mnt/data/amoskovtsev/defect-detection/')-> None:

  model = YOLO(weights)
  if torch.cuda.is_available():
    device = 'cuda:0'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  model.predict(source = inf_dir,
                imgsz = imgsz,
                conf = conf,
                save = save,
                device = device,
                project = project)

if __name__ == "__main__":
  fire.Fire(main)