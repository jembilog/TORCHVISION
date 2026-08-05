#for dataset
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

#image settings
IMAGE_SIZE = 224

#training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.00001

#model
NUM_CLASSES = 9

#save model
MODEL_PATH = "best_model.pth"

#device
DEVICE = "cuda"

SEED = 42

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PIN_MEMORY = True

DROPOUT = 0.5

SAVE_BEST = True

PATIENCE = 5
