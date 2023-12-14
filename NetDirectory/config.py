import torch


BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 40 # Number of epochs to train for.
NUM_WORKERS = 2 # Number of parallel workers for data loading.
LEARNING_RATE = 0.007475
STEP = 10
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR = '/content/Defect-Detection/NetDirectory/custom_data/train'
# Validation images and XML files directory.
VALID_DIR = '/content/Defect-Detection/NetDirectory/custom_data/valid'
TEST_DIR = '/content/Defect-Detection/NetDirectory/custom_data/test'



CLASSES = [
    '__background__', 'bridge', 'gap', 'sraf'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = '/content/Defect-Detection/NetDirectory/outputs'
