from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np
from os import path
from tqdm import tqdm

from grammar.state_enumerator import State
from attack import utils
from training.one_cycle_lr import OneCycleLR

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn import preprocessing

import psutil
import os

class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.features = self.hp.TRAIN_DATA
        self.labels = self.hp.TRAIN_LABELS
        self.test_features = self.hp.TEST_DATA
        self.test_labels = self.hp.TEST_LABELS
        self.validation_features = self.hp.VAL_DATA
        self.validation_labels = self.hp.VAL_LABELS
        self.best_snr_val_features = self.hp.BEST_SNR_VAL_DATA
        self.best_snr_val_labels = self.hp.BEST_SNR_VAL_LABELS

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list, lr):
        _optimizer = Adam(lr = lr)
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

        model = tf.keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, 
                      loss=loss, 
                      metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    def train_and_predict(self, model, parallel_no=1):
        process = psutil.Process(os.getpid())
        print(f"Memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
        model.fit(
            x=self.features,
            y=self.labels,
            batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no,
            epochs=self.hp.MAX_EPOCHS,
            validation_data=(self.validation_features, 
                             self.validation_labels),
            verbose=2
        ) 

        return (
            model.predict(self.test_features),
            model.evaluate(x=self.validation_features, y=self.validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE),
            model.evaluate(x=self.best_snr_val_features, y=self.best_snr_val_labels, batch_size=self.hp.EVAL_BATCH_SIZE)
        )