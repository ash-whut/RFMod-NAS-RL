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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn import preprocessing


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

        self.scaler_ch1 = preprocessing.StandardScaler()
        self.scaler_ch2 = preprocessing.StandardScaler()
        self.features = self.scale_data(self.features, train_data=True)
        self.test_features = self.scale_data(self.test_features, train_data=False)
        self.validation_features = self.scale_data(self.validation_features, train_data=False)
        print(self.features)

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list):
        _optimizer = Adam()  # Learning rate will be handled by OneCycleLR policy
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

        model = tf.keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, loss=loss, metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    @staticmethod
    def get_strategy():
        return tf.distribute.MirroredStrategy()

    def train_and_predict(self, model, parallel_no=1):
        print("Training started")
        print("Training labels shape: ", self.labels.shape)
        print("Validation labels shape", self.validation_labels.shape)
        model.fit(
            x=self.features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], 
            y=self.labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN], epochs=self.hp.MAX_EPOCHS,
            batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no,
            validation_data=(self.validation_features[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL], 
                             self.validation_labels[:self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL]), shuffle=True, callbacks=[
                OneCycleLR(
                    max_lr=self.hp.MAX_LR * parallel_no, end_percentage=0.2, scale_percentage=0.1,
                    maximum_momentum=None,
                    minimum_momentum=None, verbose=True
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor = 'loss',
                    patience = 5,
                    mode='auto',
                    verbose = 1)
            ]
        ) 
        print("Training finished")

        return (
            model.predict(self.test_features),
            model.evaluate(x=self.validation_features, y=self.validation_labels, batch_size=self.hp.EVAL_BATCH_SIZE)
        )
        
    def scale_data(self, data, train_data: bool):
        N, H, C = data.shape
        standardized_data = np.empty_like(data, dtype=np.float32)  # Pre-allocate output array
        scalers = [self.scaler_ch1, self.scaler_ch2]

        batch_size = 1000

        for channel in range(C):
            scaler = scalers[channel]
            for i in tqdm(range(0, N, batch_size), desc=f"Scaling train data channel {channel + 1}"):
                batch = data[i:min(i+batch_size, N), :, channel]
                batch_reshaped = batch.reshape(-1, 1)
                batch_scaled = scaler.fit_transform(batch_reshaped) if train_data else scaler.transform(batch_reshaped)
                standardized_data[i:min(i+batch_size, N), :, channel] = batch_scaled.reshape(batch.shape)
        
        return standardized_data