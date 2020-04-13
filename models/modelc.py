import torch.nn as nn
import torch
import numpy as np


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257


def compute_output_size(input_length, window, stride):
    return int((input_length - window)/stride)+1


class ModelC(nn.Module):
    def __init__(self,input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH):
        super().__init__()
        embedding_size = 16
        self.embed = nn.Embedding(input_height, embedding_size) 
        self.conv_1 = nn.Conv1d(embedding_size, 128, 16, 16)  
        self.pooling = nn.MaxPool1d(4)
        output_size = compute_output_size(input_length, 16, 16) // 4
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv_2 = nn.Conv1d(128, 128, 16, 16)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool1d(4)
        output_size = compute_output_size(output_size, 16, 16) // 4 
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.conv_3 = nn.Conv1d(128, 128, 4, 2)
        self.relu3 = nn.ReLU()
        output_size = compute_output_size(output_size, 4, 2)
        self.pooling3 = nn.MaxPool1d(output_size)
        self.fc_1 = nn.Linear(128,128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        #self.dropout = nn.Dropout(.25)
        self.fc_2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()        

    def forward(self,x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        x = self.pooling(nn.functional.relu(self.conv_1(x)))
        x = self.batch_norm1(x)
        x = self.pooling2(nn.functional.relu(self.conv_2(x)))
        x = self.batch_norm2(x)
        x = nn.functional.relu(self.conv_3(x))
        x = self.pooling3(x)
        x = x.view(-1,128)
        x = nn.functional.relu(self.fc_1(x))
        x = self.batch_norm3(x)
        #x = self.dropout(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return x