import torch
import os 
import glob as glob 
import time
from config import (BATCH_SIZE, RESIZE_TO, NUM_WORKERS,
                    DEVICE, VALID_DIR, TEST_DIR, CLASSES,
                    NUM_CLASSES, OUT_DIR, COLORS
                    )
from eval import validate
import cv2
import numpy as np
from google.colab.patches import cv2_imshow



def test_inference(DIR_TEST, CLASSES, model, CONF_THRESHOLD = 0.5): 
  
  os.makedirs('inference_outputs/images', exist_ok=True)
  test_images = glob.glob(f"{DIR_TEST}/*.jpg")
  print(f"Test instances: {len(test_images)}")
  
  
  frame_count = 0 # To count total frames.
  total_fps = 0 # To get the final frames per second.

  for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    print(image.shape)
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # Convert to tensor.
    image_input = torch.tensor(image_input, dtype=torch.float).cuda()
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)
    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Total FPS till current frame.
    total_fps += fps
    frame_count += 1

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= CONF_TRESHOLD].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            # Recale boxes.
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            cv2.rectangle(orig_image,
                        (xmin, ymin),
                        (xmax, ymax),
                        color[::-1],
                        3)
            cv2.putText(orig_image,
                        class_name,
                        (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        1,
                        lineType=cv2.LINE_AA)

        cv2_imshow(orig_image)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)
  
  print('TEST PREDICTIONS COMPLETE')
  # Calculate and print the average FPS.
  avg_fps = total_fps / frame_count
  print(f"Average FPS: {avg_fps:.3f}")


def per_class_stat(valid_loader, model, classes, save_path, metric):

  stats = validate(valid_loader, model, metric)
  print('\n')
  print(stats)

  print('\n')
  print(f"Classes: {classes}")
  print('\n')
  print('AP / AR per class')
  empty_string = ''
  if len(CLASSES) > 2:
      num_hyphens = 73
      print('-'*num_hyphens)
      print(f"|    | Class{empty_string:<16}| AP{empty_string:<18}| AR{empty_string:<18}|")
      print('-'*num_hyphens)
      class_counter = 0
      for i in range(0, len(classes)-1, 1):
          class_counter += 1
          print(f"|{class_counter:<3} | {classes[i+1]:<20} | {np.array(stats['map_per_class'][i]):.3f}{empty_string:<15}| {np.array(stats['mar_100_per_class'][i]):.3f}{empty_string:<15}|")
      print('-'*num_hyphens)
      print(f"|Avg{empty_string:<23} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
  else:
      num_hyphens = 62
      print('-'*num_hyphens)
      print(f"|Class{empty_string:<10} | AP{empty_string:<18}| AR{empty_string:<18}|")
      print('-'*num_hyphens)
      print(f"|{classes[1]:<15} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
      print('-'*num_hyphens)
      print(f"|Avg{empty_string:<12} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
  
  results_table = pd.DataFrame({
    'Class': ['Bridge','GAP','SRAF'],
    'AP':[np.round(np.array(stats['map_per_class'][i]),3) for i in range(len(classes)-1)],
    'AR':[np.round(np.array(stats['mar_100_per_class'][i]),3) for i in range(len(classes)-1)]
})

  results_table.loc[len(results_table.index)] = ['AVG', np.round(np.array(stats['map']),3 ), np.round(np.array(stats['mar_100']),3 )  ]
  results_table.to_csv(save_path)
    
