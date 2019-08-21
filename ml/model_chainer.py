""" Models implementation.
"""
import os
from pathlib import Path
import time

import chainer
from chainer import serializers
from chainer import Variable
import chainer.backends as B
import chainer.functions as F
from chainer.initializers.uniform import GlorotUniform
import chainer.links as L
from chainer.optimizers import Adam
import numpy as np

from ml.load_data import get_specific_data
from ml.utils import default_config_if_none
from utils.metrics import calc_acc


class Model(chainer.Chain):
    """Class that provides that wraps a Chainer-based neural network model
        and provides an interface for its evaluation.
    """
    def __init__(self, L_conv, D, L_fc, N, in_dim=None, learning_config=None):
        """Inits Model with corresponding paramaters.

        Args:
            L_conv: number of convolutional layers
                if any, 0 otherwise.
            D: number of filters in each convolutional layer
                if any, 0 otherwise.
            L_fc: number of fully-connected layers.
            N: number of neurons in each fully-connected layer.
        """
        super().__init__()

        self.learning_config = default_config_if_none(learning_config)
        seed = self.learning_config['seed']
        np.random.seed(seed)

        KERNEL_SIZE = 3

        self.conv_layers = []
        for c in range(L_conv):
            in_ch = 1 if c == 0 else D
            out_ch = D
            conv = L.Convolution2D(
                in_ch, out_ch,
                KERNEL_SIZE, pad=(KERNEL_SIZE - 1)//2,
                initialW=GlorotUniform()
            )
            self.add_link('conv' + str(c), conv)
            self.conv_layers.append(conv)

        self.fc_in_dim = 15*D if L_conv > 0 else in_dim
        self.has_conv = L_conv > 0

        self.fc_layers = []
        for c in range(L_fc):
            in_ch = self.fc_in_dim if c == 0 else N
            out_ch = N
            fc = L.Linear(in_ch, out_ch, initialW=GlorotUniform())
            self.add_link('fc' + str(c), fc)
            self.fc_layers.append(fc)

        out_fc = L.Linear(N, 2, initialW=GlorotUniform())
        self.add_link('out', out_fc)
        self.fc_layers.append(out_fc)

        try:
            import cupy as cp
            B.cuda.check_cuda_available()
            chainer.global_config.cudnn_deterministic = True
            self.dev = '@cupy:0'
            cp.random.seed(seed)
            self.random_permutation = np.random.permutation
        except:
            print("Chainer: CUDA isn't available, using CPU...")
            self.dev = '@numpy'
            self.random_permutation = np.random.permutation
        
        self.to_device(self.dev)

        self.opt = Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
        self.opt.setup(self)
        self.opt.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001))

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        if len(self.conv_layers) > 0:
            x = F.reshape(x, (-1, self.fc_in_dim))
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        return self.fc_layers[-1](x)

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

        N = X.shape[0]
        X = self._tensor_from_numpy(X)
        y = self._tensor_from_numpy(y)

        X_val = self._tensor_from_numpy(X_val)
        y_val = self._tensor_from_numpy(y_val)

        crit = F.mean_squared_error
        fit_time = time.time()

        for epoch in range(epochs):
            epoch_ind = self.random_permutation(N)
            extra_step = 0 if N % batch_size == 0 else 1
            for i in range(N // batch_size + extra_step):
                batch_ind = epoch_ind[i*batch_size: (i + 1)*batch_size]
                batch_X, batch_y_gt = X[batch_ind, :], y[batch_ind, :]
                batch_y_pred = self(batch_X)

                loss = crit(batch_y_pred, batch_y_gt)

                self.cleargrads()
                loss.backward()
                self.opt.update()

            # train_loss = crit(self(X), y).array.item()
            # val_loss = crit(self(X_val), y_val).array.item()

            # epoch_str = 'Epoch: {:>5};'.format(epoch)
            # loss_str = 'Train loss: {:>7.3f}; Val loss: {:>7.3f}'.format(
            #     train_loss, val_loss
            # )
            # print(epoch_str, loss_str)
    
        return time.time() - fit_time

    def _tensor_from_numpy(self, x):
        var = Variable(x.astype(np.float32), requires_grad=False)
        var.to_device(self.dev)
        return var

    def _numpy_from_tensor(self, x):
        x.to_device(chainer.get_device('@numpy'))
        return x.array

    def _predict(self, X, y):
        X = self._tensor_from_numpy(X)
        y_pred = self(X)
        # print('Loss: ', F.mean_squared_error(y_pred, self._tensor_from_numpy(y)))
        acc = calc_acc(y, self._numpy_from_tensor(y_pred))
        return acc

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
        train_acc = self._predict(X_train, y_train)
        test_acc = self._predict(X_test, y_test)

        val_acc = None
        if X_val is not None and y_val is not None:
            val_acc = self._predict(X_val, y_val)

        return train_acc, test_acc, val_acc

    def _add_impl_prefix(self, model_path):
        model_dir = str(Path(model_path).parent)
        model_name = 'chainer_' + str(Path(model_path).name)
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
        serializers.save_hdf5(model_path, self)

    def load_weights(self, model_path):
        """Load model's weights.

        Args:
            model_path: A string with full path for model to be loaded from.
        """
        model_path = self._add_impl_prefix(model_path)
        serializers.load_hdf5(model_path, self)

    def freeze_conv(self):
        """Freeze weights for convolutional layers of the self.
        """
        for conv in self.conv_layers:
            conv.W.requires_grad = False