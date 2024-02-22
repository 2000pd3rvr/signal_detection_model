import torch
import os


BASE_PATH = "/Users/pd3rvr/Documents/rbkp2/models/signal_detection_model/dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])


BASE_OUTPUT = "/Users/pd3rvr/Documents/rbkp2/models/signal_detection_model/dataset/output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"]) #label encoder
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


# determine the current device and based on that set the pin memory
# flag
# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



print(f"Using {DEVICE} device")

PIN_MEMORY = True if DEVICE == "CPU" or DEVICE == "mps" or DEVICE == "cuda" else False
# specify mean and standard deviation for channel-wise, width-wise, and height-wise mean and standard deviation, respectively.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
# specify the loss weights
LABELS = 4.0
BBOX = 1.0
