from ultralytics import YOLO
import wandb
import fire
import torch

def main(model_config:str = None,
         model_type: str ='yolo11s.pt',
         data_config: str ='data/processed/defects/data.yaml',
         project: str ='/mnt/data/amoskovtsev/defect-detection',
         epochs: int = 50,
         imgsz: int = 640,
         batch: int = 16,
         workers: int = 8,
         seed:int = 42,
         deterministic: bool = True,
         log_wandb: bool = True) -> None:

  if torch.cuda.is_available():
    device = 'cuda:0'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  if log_wandb:
    WANDB_KEY = '7f7117ef2660f827c823ba03863048fe0eea4801'
    wandb.login(WANDB_KEY)
    model = YOLO(model_type)
    print(f'Model info:{model.info()}')

    results = model.train(
      model = model_config,
      data = data_config,
      epochs = epochs,
      batch = batch,
      imgsz = imgsz,
      workers = workers,
      project = 'defect-detection',
      seed = seed,
      deterministic = deterministic,
      device = device
    )
  else:
    model = YOLO(model_type)
    print(f'Model info:{model.info()}')
    results = model.train(
      model = model_config,
      data = data_config,
      epochs = epochs,
      batch = batch,
      imgsz = imgsz,
      workers = workers,
      project = project,
      seed = seed,
      deterministic = deterministic,
      device = device
    )

  print(f'Training results:{results}')
  metrics = model.val(data=data_config)
  print(f'Metrics on validation set: {metrics}')


if __name__ == "__main__":
  fire.Fire(main)