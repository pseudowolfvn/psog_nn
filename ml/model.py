""" Models implementation.
"""
import os
from pathlib import Path
import time

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from utils.metrics import calc_acc
from utils.utils import get_arch


class Model(nn.Module):
    """Class that provides that wraps a Keras-based neural network model
        and provides an interface for its evaluation.
    """
    def __init__(self, L_conv, D, L_fc, N, in_dim=None):
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

        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

        KERNEL_SIZE = 3

        for c in range(L_conv):
            in_ch = 1 if c == 0 else D
            out_ch = D
            self.add_module(
                'conv' + str(c),
                nn.Conv2d(
                    in_ch, out_ch,
                    KERNEL_SIZE, padding=(KERNEL_SIZE - 1)//2
                )
            )

        self.fc_in_dim = 15*D if L_conv > 0 else in_dim
        self.has_conv = L_conv > 0

        for c in range(L_fc):
            in_ch = self.fc_in_dim if c == 0 else N
            out_ch = N
            self.add_module(
                'fc' + str(c),
                nn.Linear(in_ch, out_ch)
            )

        self.add_module(
            'out',
            nn.Linear(N, 2)
        )

        self.to(self.device)

    def forward(self, x):
        flattened = False
        for name, l in self.named_children():
            if not flattened and self.has_conv and name.startswith('fc'):
                x = x.view(-1, self.fc_in_dim)
                flattened = True
            x = F.relu(l(x)) if not name.startswith('out') else l(x)

        return x

    def fit(self, X, y, X_val, y_val,
            epochs=1000, batch_size=200, patience=100):
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
        opt = Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        crit = nn.MSELoss()

        N = X.shape[0]
        X = self._tensor_from_numpy(X)
        y = self._tensor_from_numpy(y)

        X_val = self._tensor_from_numpy(X_val)
        y_val = self._tensor_from_numpy(y_val)

        fit_time = time.time()

        counter = 0
        best_loss = None

        # TEMP CODE
        for epoch in range(epochs):
            epoch_ind = torch.randperm(N)
            for i in range(N // batch_size):
                batch_ind = epoch_ind[i*batch_size: (i + 1)*batch_size]
                batch_X, batch_y_gt = X[batch_ind, :], y[batch_ind, :]
                batch_y_pred = self(batch_X)

                opt.zero_grad()
                loss = crit(batch_y_pred, batch_y_gt)

                loss.backward()
                opt.step()

            train_loss = crit(self(X), y).item()
            val_loss = crit(self(X_val), y_val).item()

            epoch_str = 'Epoch: {:>5};'.format(epoch)
            loss_str = 'Train loss: {:>7.3f}; Val loss: {:>7.3f}'.format(train_loss, val_loss)
            print(epoch_str, loss_str)

            # Early stopping
            if best_loss is None:
                best_loss = val_loss
            elif val_loss >= best_loss:
                counter += 1
                if counter >= patience:
                    print('Early stopping triggered')
                    break
            else:
                best_loss = val_loss
                counter = 0

        return time.time() - fit_time

    def _tensor_from_numpy(self, X):
        return torch.tensor(X).float().to(self.device)

    def _numpy_from_tensor(self, X):
        return X.detach().cpu().numpy()

    def _predict(self, X, y):
        X = self._tensor_from_numpy(X)
        y_pred = self(X)
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

    def save_weights(self, model_path):
        """Save model's weights.

        Args:
            model_path: A string with full path for model to be saved.
        """
        model_dir = str(Path(model_path).parent)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(self.state_dict(), model_path)

    def load_weights(self, model_path):
        """Load model's weights.

        Args:
            model_path: A string with full path for model to be loaded from.
        """
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def freeze_conv(self):
        """Freeze weights for convolutional layers of the self.
        """
        for name, layer in self.named_children():
            if name.startswith('conv'):
                layer.requires_grad = False

CNN = Model

class MLP(Model):
    def __init__(self, layers, neurons, in_dim):
        super().__init__(0, 0, layers, neurons, in_dim)

# TODO: rewrite to factory
def build_model(params, in_dim=None):
    """The interface that should be used to obtain the instance of
        Model class with provided parameters.
    
    Args:
        params: A tuple with neural network paramters 
            with the following format: (
                <number of convolutional layers
                    if any, 0 otherwise>,
                <number of filters in each convolutional layer
                    if any, 0 otherwise>,
                <number of fully-connected layers>,
                <number of neurons in each fully-connected layer>
            ).

    Returns:
        An instance of Model class that represents
            a model with corresponding parameters.
    """
    arch = get_arch(params) 
    if arch == 'mlp':
        return MLP(*params[-2:], in_dim)
    if arch == 'cnn':
        return CNN(*params)
    return None
