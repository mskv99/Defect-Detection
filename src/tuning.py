from ultralytics import YOLO
import fire
import wandb
import torch
def main(model_type: str = 'yolo11s.pt',
         data_config: str = 'data/processed/dataset-v1/data.yaml',
         epochs: int = 20,
         iterations: int = 15,
         optimizer: str = "AdamW",
         plots: bool = False,
         save: bool = False,
         val: bool = False,
         log_wandb: bool = True
         ):
  if torch.cuda.is_available():
    device = 'cuda:0'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'
  print(f'Tuning hyperparameters on {device} device...')

  model = YOLO(model_type)

  if log_wandb:
    WANDB_KEY = '7f7117ef2660f827c823ba03863048fe0eea4801'
    wandb.login(key=WANDB_KEY)
    model.tune(
      data=data_config,
      epochs=epochs,
      iterations=iterations,
      optimizer=optimizer,
      plots=plots,
      save=save,
      val=val,
    )
  else:
    model.tune(
      data=data_config,
      epochs=epochs,
      iterations=iterations,
      optimizer=optimizer,
      plots=plots,
      save=save,
      val=val,
    )

if __name__ == "__main__":
  fire.Fire(main)