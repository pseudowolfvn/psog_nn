""" Models implementation.
"""
import os
from pathlib import Path
import time

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

from ml.utils import default_config_if_none
from utils.metrics import calc_acc


class Model:
    """Class that provides that wraps a Keras-based neural network model
        and provides an interface for its evaluation.
    """
    def __init__(self, L_conv, D, L_fc, N, learning_config=None):
        """Inits Model with corresponding paramaters.

        Args:
            L_conv: number of convolutional layers
                if any, 0 otherwise.
            D: number of filters in each convolutional layer
                if any, 0 otherwise.
            L_fc: number of fully-connected layers.
            N: number of neurons in each fully-connected layer.
        """
        self.learning_config = default_config_if_none(learning_config)

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

    def fit(self, X, y, X_val, y_val):
        """Train the model.

        Args:
            X: An array of input values with
                PSOG sensor raw outputs.
            y: An array of output values with
                corresponding ground-truth eye gazes.
            X_val: An array with validation set for input.
            y_val: An array with validation set for output.
            epochs: An int with number of epochs to train for.
            batch_size: An int with batch size.
            patience: An int with the number of epochs to wait for
                validation loss improvement in early-stopping technique.

        Returns:
            A float with time spent for training in sec.
        """
        epochs = self.learning_config['epochs']
        batch_size = self.learning_config['batch_size']
        patience = self.learning_config['patience']

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience,
            mode='auto', restore_best_weights=True
        )

        # Keras doesn't save optimizer together with the model
        nadam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
        """Calculate accuracy of model's predictions.

        Args:
            X_train: An array of training set input values with
                PSOG sensor raw outputs.
            y_train: An array of training set output values with
                corresponding ground-truth eye gazes.
            X_test: An array with test set for input.
            y_test: An array with test set for output.
            X_val: An array with validation set for input.
            y_val: An array with validation set for output.

        Retuns:
            A tuple with spatial accuracies on training set, test set and
                validation set.
        """
        train_acc = calc_acc(y_train, self.model.predict(X_train))
        test_acc = calc_acc(y_test, self.model.predict(X_test))
        val_acc = None
        if X_val is not None and y_val is not None:
            val_acc = calc_acc(y_val, self.model.predict(X_val))
        return train_acc, test_acc, val_acc

    def _add_impl_prefix(model_path):
        model_dir = str(Path(model_path).parent)
        model_name = 'keras_' + str(Path(model_path).name)
        return os.path.join(model_dir, model_name)

    def save_weights(self, model_path):
        """Save model's weights.

        Args:
            model_path: A string with full path for model to be saved.
        """
        model_dir = str(Path(model_path).parent)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_path = self._add_impl_prefix(model_path)
        self.model.save(model_path)

    def load_weights(self, model_path):
        """Load model's weights.

        Args:
            model_path: A string with full path for model to be loaded from.
        """
        model_path = self._add_impl_prefix(model_path)
        self.model = load_model(model_path)

    def freeze_conv(self):
        """Freeze weights for convolutional layers of the self.
        """
        for layer in self.model.layers:
            if layer.name.startswith('conv2d'):
                layer.trainable = False
