import torch.nn as nn
import torch
import numpy as np


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257


class ModelC(nn.Module):
    def __init__(self,input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH):
        super().__init__()
        embedding_size = 16
        self.embed = nn.Embedding(input_height, embedding_size) 
        self.conv_1 = nn.Conv1d(embedding_size, 20, 16, 4)  
        self.pooling = nn.MaxPool1d(4)
        output_size = int(compute_output_size(input_length, 16, 4) / 4)
        self.conv_2 = nn.Conv1d(20, 40, 16, 4)
        self.relu2 = nn.ReLU()
        self.pooling = nn.MaxPool1d(4)
        output_size = int(compute_output_size(output_size, 16, 4) / 4)  
        self.conv_3 = nn.Conv1d(40, 80, 4, 2)
        self.relu3 = nn.ReLU()
        output_size = compute_output_size(output_size, 4, 2)
        self.pooling2 = nn.MaxPool1d(output_size)
        self.fc_1 = nn.Linear(80,80)
        self.fc_2 = nn.Linear(80,1)
        self.sigmoid = nn.Sigmoid()        

    def forward(self,x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        x = self.pooling(nn.functional.relu(self.conv_1(x)))
        x = self.pooling(nn.functional.relu(self.conv_2(x)))
        x = nn.functional.relu(self.conv_3(x))
        x = self.pooling2(x)
        x = x.view(-1,80)
        x = nn.functional.relu(self.fc_1(x))
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return x
