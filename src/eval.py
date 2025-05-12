from ultralytics import YOLO
import fire
import torch

def main(data_config: str = 'data/processed/defects/data.yaml',
         weights: str = None,
         project: str = '/mnt/data/amoskovtsev/defect-detection/'):
  if torch.cuda.is_available():
    device = 'cuda:0'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  model = YOLO(weights)
  metrics = model.val(data=data_config,
                      project = project,
                      device = device)
  print(f'Metrics: {metrics}')

if __name__=='__main__':
  fire.Fire(main)

