from metaqnn.data_loader import *
from . import state_space_parameters as ssp
import numpy as np

MODEL_NAME = 'RML_2021'
DATASET_PATH = '/home/ashwin/datasets/RADIOML_2021_07_INT8/RADIOML_2021_07_INT8.hdf5'

# Number of output neurons
NUM_CLASSES = 27  # Number of output neurons

# Input Size
INPUT_SIZE = 2200

# Batch Queue parameters
TRAIN_BATCH_SIZE = 128  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45000  # Number of training examples
VALIDATION_FROM_ATTACK_SET = True
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2500  # Number of validation examples


TEST_INTERVAL_EPOCHS = 1  # Num epochs to test on, should really always be 1
MAX_EPOCHS = 50  # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 5e-3  # The max LR (scaled linearly with number of gpus used)

# Reward small parameter
# This rewards networks smaller than this number of trainable parameters
MAX_TRAINABLE_PARAMS_FOR_REWARD = 142044

TRAIN_DATA, TRAIN_LABELS, TRAIN_SNRS, TEST_DATA, TEST_LABELS, TEST_SNRS =  load_data(DATASET_PATH)
VAL_DATA, VAL_LABELS, UNUSED_TEST_DATA, UNUSED_TEST_LABELS = validation_data(TEST_DATA, TEST_LABELS)

TRACES_PER_ATTACK = 2000  # Maximum number of traces to use per attack
NUM_ATTACKS = 100  # Number of attacks to average the GE over
