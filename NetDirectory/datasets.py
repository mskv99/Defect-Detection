from xml.etree import ElementTree as et
import random
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob as glob
from google.colab.patches import cv2_imshow

from config import (BATCH_SIZE, RESIZE_TO, NUM_WORKERS,
                    DEVICE,TRAIN_DIR, VALID_DIR, CLASSES,
                    NUM_CLASSES, OUT_DIR, COLORS
                    )


class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]

        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        #image_resized /= 255.0

        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            labels.append(self.classes.index(member.find('name').text))

            # Left corner x-coordinates.
            xmin = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            xmax = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            ymin = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            ymax = int(member.find('bndbox').find('ymax').text)

            # Resize the bounding boxes according
            # to resized image `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            # Check that all coordinates are within the image.
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def plot_box(image, target):
    # Need the image height and width to denormalize

    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.

    pred_classes = [CLASSES[i] for i in target['labels']]
    for box_num in range(len(target['boxes'])):
      box = target['boxes'][box_num]
      class_name = pred_classes[box_num]
      color=COLORS[CLASSES.index(class_name)]

      p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))



      cv2.rectangle(
          image,
          p1, p2,
          color=(26,26,139),
          thickness=lw,
          lineType=cv2.LINE_AA
      )

      # For filled rectangle.
      w, h = cv2.getTextSize(
          class_name,
          0,
          fontScale=lw / 3,
          thickness=tf
      )[0]

      outside = p1[1] - h >= 3
      p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

      cv2.rectangle(
          image,
          p1, p2,
          color=(26,26,139),
          thickness=-1,
          lineType=cv2.LINE_AA
      )
      cv2.putText(
          image,
          class_name,
          (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
          cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=lw/3.5,
          color=(255, 255, 255),
          thickness=tf,
          lineType=cv2.LINE_AA
      )
    return image

def show_tranformed_image(train_loader, device, classes, colors):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    """
    if len(train_loader) > 0:
        for i in range(BATCH_SIZE):
            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            # Get all the predicited class names.
            pred_classes = [classes[i] for i in targets[i]['labels'].cpu().numpy()]
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

            lw = max(round(sum(sample.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.


            for box_num, box in enumerate(boxes):

              class_name = pred_classes[box_num]
              color=colors[classes.index(class_name)]

              p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))


              cv2.rectangle(
                  sample,
                  p1, p2,
                  color=(26,26,139),
                  thickness=lw,
                  lineType=cv2.LINE_AA
              )

              # For filled rectangle.
              w, h = cv2.getTextSize(
                  class_name,
                  0,
                  fontScale=lw / 3,
                  thickness=tf
              )[0]

              outside = p1[1] - h >= 3
              p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

              cv2.rectangle(
                  sample,
                  p1, p2,
                  color=(26,26,139),
                  thickness=-1,
                  lineType=cv2.LINE_AA
              )
              cv2.putText(
                  sample,
                  class_name,
                  (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=lw/3.5,
                  color=(255, 255, 255),
                  thickness=tf,
                  lineType=cv2.LINE_AA
              )
            cv2_imshow(sample)
            print('\n')
