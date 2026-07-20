import torch
import torch.nn as nn


class MyNN(nn.Module):
    def __init__(self, input, output):
        super().__init__()

        self.layer_1 = nn.Linear(input,128)
        self.batch_norm_1 = nn.BatchNorm1d(128)
        self.act_1 = nn.LeakyReLU()
        self.dropout_1 = nn.Dropout(0.4)

        self.layer_2 = nn.Linear(128,64)
        self.batch_norm_2 = nn.BatchNorm1d(64)
        self.act_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.2)

        self.layer_3 = nn.Linear(64,output)

    def forward(self,X):
        X = self.layer_1(X)
        X = self.batch_norm_1(X)
        X = self.act_1(X)
        X = self.dropout_1(X)

        X = self.layer_2(X)
        X = self.batch_norm_2(X)
        X = self.act_2(X)
        X = self.dropout_2(X)

        out = self.layer_3(X)
        return out