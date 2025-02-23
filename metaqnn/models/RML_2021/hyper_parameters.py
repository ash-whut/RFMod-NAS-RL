from data_loader import *
import numpy as np

MODEL_NAME = 'RML_2021'
DATASET_PATH = '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'

# Number of output neurons
NUM_CLASSES = 27  # Number of output neurons

# Input Size
INPUT_SIZE = 1024

# Batch Queue parameters
TRAIN_BATCH_SIZE = 128  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000  # Number of training examples
VALIDATION_FROM_ATTACK_SET = True
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000  # Number of validation examples


TEST_INTERVAL_EPOCHS = 1  # Num epochs to test on, should really always be 1
MAX_EPOCHS = 2  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 5e-3  # The max LR (scaled linearly with number of gpus used)

# Reward small parameter
# This rewards networks smaller than this number of trainable parameters
MAX_TRAINABLE_PARAMS_FOR_REWARD = 65000

TRAIN_DATA, TEST_DATA, TRAIN_LABELS, TEST_LABELS, TRAIN_SNRS, TEST_SNRS =  load_data(DATASET_PATH)
VAL_DATA, UNUSED_TEST_DATA, VAL_LABELS, UNUSED_TEST_LABELS = validation_data(TEST_DATA, TEST_LABELS)

TRAINED_MODEL_DIR = '../trained_models/'