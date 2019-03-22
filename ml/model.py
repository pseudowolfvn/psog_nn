""" Models implementation.
"""
import os
from pathlib import Path
import time

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Nadam
from keras.regularizers import l2

from utils.metrics import calc_acc
from utils.utils import get_arch


class Model:
    def __init__(self, L_conv, D, L_fc, N):
        self.model = Sequential()

        for _ in range(L_conv):
            self.model.add(Conv2D(D, 3, padding='same'))

        if L_conv > 0:
            self.model.add(Flatten())

        for _ in range(L_fc):
            self.model.add(Dense(
                N, activation='relu',
                kernel_regularizer=l2(0.0001)
            ))

        self.model.add(Dense(2))

    def train(self, X, y, X_val, y_val,
            epochs=1000, batch_size=200, patience=100):
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience,
            mode='auto', restore_best_weights=True
        )

        # Keras doesn't save optimizer together with the model
        nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='mean_squared_error', optimizer=nadam_opt)

        fit_time = time.time()
        self.model.fit(
            X, y, nb_epoch=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val), verbose=1,
            callbacks=[early_stopping]
        )
        return time.time() - fit_time

    def report_acc(self, X_train, y_train,
            X_test, y_test, X_val=None, y_val=None):
        train_acc = calc_acc(y_train, self.model.predict(X_train))
        test_acc = calc_acc(y_test, self.model.predict(X_test))
        val_acc = None
        if X_val is not None and y_val is not None:
            val_acc = calc_acc(y_val, self.model.predict(X_val))
        return train_acc, test_acc, val_acc

    def save_weights(self, model_path):
        model_dir = str(Path(model_path).parent)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.model.save(model_path)

    def load_weights(self, model_path):
        self.model = load_model(model_path)

    def freeze_conv(self):
        for layer in self.model.layers:
            if layer.name.startswith('conv2d'):
                layer.trainable = False

CNN = Model

class MLP(Model):
    def __init__(self, layers, neurons):
        super().__init__(0, 0, layers, neurons)

# TODO: rewrite to factory
def build_model(params):
    K.clear_session()
    arch = get_arch(params) 
    if arch == 'mlp':
        return MLP(*params[-2:])
    if arch == 'cnn':
        return CNN(*params)
    return None
