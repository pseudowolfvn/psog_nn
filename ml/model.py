import time

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.regularizers import l2

from utils.metrics import calc_acc


class Model(Sequential):
    def __init__(self, conv_layers, conv_kernel, layers, neurons)
        for _ in range(conv_layers):
            self.add(Conv2D(conv_kernel, 3, padding='same'))

        if conv_layers > 0:
            self.add(Flatten())

        for _ in range(layers):
            self.add(Dense(neurons, activation='relu', kernel_regularizer=l2(0.0001)))

        self.add(Dense(2))

        nadam_opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.compile(loss='mean_squared_error', optimizer=nadam_opt)

    def train(X, y, X_val, y_val, epochs=1000, batch_size=200, patience=100):
        early_stopping = EarlyStopping(monitor='val_loss'
            , patience=patience, mode='auto', restore_best_weights=True)

        fit_time = time.time()
        self.fit(X, y, nb_epoch=epochs, batch_size=batch_size
            , validation_data=(X_val, y_val), verbose=1
            , callbacks=[early_stopping])
        return time.time() - fit_time
        
    def report_acc(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        train_acc = calc_acc(y_train, model.predict(X_train))
        test_acc = calc_acc(y_test, model.predict(X_test))
        val_acc = None
        if X_val is not None and y_val is not None:
            val_acc = calc_acc(y_val, model.predict(X_val))
        return train_acc, test_acc, val_acc

ModelCNN = Model

class ModelMLP(Model):
    def __init__(self, layers, neurons):
        super().__init__(0, 0, layers, neurons)