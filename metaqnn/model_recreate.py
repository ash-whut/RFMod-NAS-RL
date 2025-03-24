import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc

import models.RML_2021.hyper_parameters as hp
import models.RML_2021.state_space_parameters as ssp
import grammar.cnn
from grammar.state_string_utils import StateStringUtils
from training.tensorflow_runner import TensorFlowRunner

top_model_strings = pd.read_csv("learner_logs/replay_database.csv").sort_values(by=["accuracy"], ascending = False).head(1)["net"].values

def model_generator(net_strings):
    models = []
    for string in net_strings:
        stt_str_utls = StateStringUtils(ssp)
        cnn_parsed = grammar.cnn.parse("net", string)
        states = stt_str_utls.convert_model_string_to_states(cnn_parsed)
            
        tfr = TensorFlowRunner(ssp, hp)
        model = tfr.compile_model(states, loss='categorical_crossentropy', metric_list=['accuracy'], lr=1e-4)
        models.append(model)
    return models

def model_trainer(models):
    trained_models = []
    for model in models:
            print(model.summary())
            model.fit(
                x=hp.TRAIN_DATA,
                y=hp.TRAIN_LABELS,
                batch_size=hp.TRAIN_BATCH_SIZE,
                epochs=100,
                validation_data=(hp.VAL_DATA, hp.VAL_LABELS),
                verbose=1,
                callbacks=[EarlyStopping(
                            monitor = 'val_loss',
                            patience = 5,
                            mode='auto',
                            verbose = 1)],
            ) 
            
            trained_models.append(model)
            K.clear_session()
            gc.collect()
            
    return trained_models

models = model_generator(top_model_strings)
print(models[0].summary())
best_trained_model = model_trainer(models)[0]

best_trained_model.save("best_trained_model.keras")