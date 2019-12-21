import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class GenericModel(nn.Module):
    def _compute_class_weights(self, y):
        fix_mask = y == 1
        sac_mask = y == 0

        N = y.shape[0]
        fix_w = 1. - y[fix_mask].shape[0] / N
        sac_w = 1. - fix_w

        # assuming :
        # class 0 = saccade
        # class 1 = fixation
        return [sac_w, fix_w]

    def fit(self, X, y, X_val, y_val, learning_config):
        self.train()

        epochs = learning_config['epochs']
        batch_size = learning_config['batch_size']
        patience = learning_config['patience']
        learning_rate = learning_config['lr']

        opt = Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.0001, eps=1e-08)
        class_weights = self._compute_class_weights(y)
        crit = nn.CrossEntropyLoss(weight=self._tensor_from_numpy(class_weights))

        N = X.shape[0]
        X = self._tensor_from_numpy(X)
        y = self._tensor_from_numpy(y)

        X_val = self._tensor_from_numpy(X_val)
        y_val = self._tensor_from_numpy(y_val)

        counter = 0
        best_loss = None
        loss_history = []

        # TEMP CODE
        for epoch in range(epochs):
            epoch_ind = torch.randperm(N, device=self.device)
            extra_step = 0 if N % batch_size == 0 else 1

            train_loss = torch.tensor(0.)

            for i in range(N // batch_size + extra_step):
                batch_ind = epoch_ind[i*batch_size: (i + 1)*batch_size]
                batch_X, batch_y_gt = X[batch_ind, :], y[batch_ind]
                batch_y_pred = self(batch_X)

                opt.zero_grad()
                loss = crit(batch_y_pred, batch_y_gt)
                train_loss += loss

                loss.backward()
                opt.step()

            # train_loss = crit(self(X), y).item()
            # Fix for memory consumption
            train_loss = train_loss / (N // batch_size + extra_step)
            train_loss = train_loss.item()

            val_loss = crit(self(X_val), y_val).item()
            loss_history.append([train_loss, val_loss])

            epoch_str = 'Epoch: {:>5};'.format(epoch)
            loss_str = 'Train loss: {:>7.3f}; Val loss: {:>7.3f}'.format(
                train_loss, val_loss
            )
            print(epoch_str, loss_str)

            # Early stopping
            if best_loss is None:
                best_loss = val_loss
            # elif val_loss >= best_loss:
            elif (best_loss - val_loss) / best_loss < 0.01:
                counter += 1
                if counter >= patience:
                    print('Early stopping triggered on epoch:', epoch)
                    print(
                        'Train loss: {:.2f}, val loss: {:.2f}, best val loss: {:.2f}'.format(
                            train_loss, val_loss, best_loss
                        )
                    )
                    break
            else:
                best_loss = val_loss
                counter = 0

        return loss_history

    def _tensor_from_numpy(self, X):
        return torch.tensor(X, device=self.device)

    def _numpy_from_tensor(self, X):
        return X.detach().cpu().numpy()

    def predict(self, X):
        self.eval()

        X = self._tensor_from_numpy(X)
        y_pred = torch.argmax(self(X), 1)

        return self._numpy_from_tensor(y_pred)

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
        # model_path = self._add_impl_prefix(model_path)
        self.load_state_dict(torch.load(model_path))
        self.eval()

class Model1DCNN(GenericModel):
    def __init__(self, ch_in, l_in, L_conv=3, D=24, L_fc=3, K=3, N=20, p_drop=0.2):
        super().__init__()

        torch.manual_seed(42)
        np.random.seed(42)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.conv_ch_in = ch_in

        self.conv_layers = []
        self.bn_conv_layers = []
        for c in range(L_conv):
            in_ch = ch_in if c == 0 else D
            out_ch = D
            conv = nn.Conv1d(in_ch, out_ch, K)

            nn.init.xavier_uniform(conv.weight)
            self.add_module('conv' + str(c), conv)
            self.conv_layers.append(conv)

            bn = nn.BatchNorm1d(out_ch)
            self.add_module('bn_conv' + str(c), bn)
            self.bn_conv_layers.append(bn)

        self.fc_in = D * (l_in - (K - 1)*L_conv)

        self.fc_layers = []
        self.bn_fc_layers = []
        for c in range(L_fc):
            in_ch = self.fc_in if c == 0 else N
            out_ch = N
            fc = nn.Linear(in_ch, out_ch)
            nn.init.xavier_uniform(fc.weight)
            self.add_module('fc' + str(c), fc)
            self.fc_layers.append(fc)

            bn = nn.BatchNorm1d(out_ch)
            self.add_module('bn_fc' + str(c), bn)
            self.bn_fc_layers.append(bn)

        out_fc = nn.Linear(N, 2)
        nn.init.xavier_uniform(out_fc.weight)
        self.add_module('out', out_fc)
        self.fc_layers.append(out_fc)

        self.drop = nn.Dropout(p=p_drop)

        self.to(self.device)

    # def forward(self, x):
    #     for conv, bn in zip(self.conv_layers, self.bn_conv_layers):
    #         x = bn(conv(self.drop(x)))
    #     x = x.view(-1, self.fc_in)
    #     for fc, bn in zip(self.fc_layers[:-1], self.bn_fc_layers):
    #         x = bn(F.relu(fc(self.drop(x))))
    #     x = self.fc_layers[-1](x)
    #     return x

    def forward(self, x):
        L_conv = len(self.conv_layers)
        L_fc = len(self.fc_layers) - 1

        for i in range(L_conv):
            conv = self.conv_layers[i]
            bn = self.bn_conv_layers[i]
            x = conv(self.drop(x))
            # x = bn(x)
        x = x.view(-1, self.fc_in)
        for i in range(L_fc):
            fc = self.fc_layers[i]
            bn = self.bn_fc_layers[i]
            x = F.relu(fc(self.drop(x)))
            # x = bn(x)
        x = self.fc_layers[-1](x)
        return x