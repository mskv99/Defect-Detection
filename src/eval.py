from ultralytics import YOLO
import fire
import torch

def main(data_config: str = 'data/processed/dataset-v1/data.yaml',
         weights: str = None,
         project: str = '/mnt/data/amoskovtsev/defect-detection/',
         conf: float = 0.001,
         iou: float = 0.6,
         split: str = 'val',
         batch: int = 16):
  if torch.cuda.is_available():
    device = 'cuda:0'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'
  print(f'Running evaluation on {device} device...') 

  model = YOLO(weights)
  metrics = model.val(data=data_config,
                      project = project,
                      device = device,
                      conf = conf,
                      iou = iou,
                      split = split,
                      batch = batch)

if __name__=='__main__':
  fire.Fire(main)

