""" Models implementation.
"""
import os
from pathlib import Path
import time

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss

from utils.metrics import calc_acc
from utils.utils import get_arch

class EyeGazeWrapper(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


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
        self.conv_layers = []
        self.fc_layers = []

        KERNEL_SIZE = 3

        for c in range(L_conv):
            in_ch = 1 if c == 0 else D
            out_ch = D
            self.conv_layers.append(
                nn.Conv2d(
                    in_ch, out_ch,
                    KERNEL_SIZE, padding=(KERNEL_SIZE - 1)//2
                )
            )
        self.fc_in_dim = 15*D if L_conv > 0 else in_dim
        for c in range(L_fc):
            in_ch = self.fc_in_dim if c == 0 else N
            out_ch = N
            self.fc_layers.append(
                nn.Linear(in_ch, out_ch)
            )

        self.fc_layers.append(
            nn.Linear(N, 2)
        )

    def forward(self, x):
        for l in self.conv_layers:
            x = F.relu(l(x))
        x = x.view(-1, self.fc_in_dim)
        for l in self.fc_layers:
            x = F.relu(l(x))
        return x

    def train(self, X, y, X_val, y_val,
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
        opt = Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        crit = nn.MSELoss()
        
        device = 'cuda'
        metrics = {
            'accuracy': Accuracy(),
            'loss': Loss(crit)
        }

        trainer = create_supervised_trainer(self, opt, crit, device=device)

        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        early_stopping = EarlyStopping(
            patience=patience,
            score_function=score_function,
            trainer=trainer
        )

        train_evaluator = create_supervised_evaluator(
            self, metrics=metrics, device=device
        )
        validation_evaluator = create_supervised_evaluator(
            self, metrics=metrics, device=device
        )

        train_loader = DataLoader(
            EyeGazeWrapper(X, y),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            EyeGazeWrapper(X_val, y_val),
            batch_size=batch_size,
            shuffle=True
        )

        fit_time = time.time()
        trainer.run(train_loader, max_epochs=epochs)
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
        train_acc = calc_acc(y_train, self.forward(X_train))
        test_acc = calc_acc(y_test, self.forward(X_test))
        val_acc = None
        if X_val is not None and y_val is not None:
            val_acc = calc_acc(y_val, self.forward(X_val))
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
        for layer in self.conv_layers:
            layer.weight.requires_grad = False

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
