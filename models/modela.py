import torch.nn as nn
import torch
import numpy as np


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257


class ModelA(nn.Module):
    def __init__(self,input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH, window_size=16, stride=4):
        super().__init__()
        embedding_size = 16
        self.embed = nn.Embedding(input_height, embedding_size) 
        self.conv_1 = nn.Conv1d(embedding_size, 128, window_size, stride, bias=True)
        self.pooling = nn.MaxPool1d(int((input_length - window_size)/stride)+1)
        self.fc_2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()        


    def forward(self,x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        x = nn.functional.relu(self.conv_1(x))
        x = self.pooling(x)
        x = x.view(-1,128)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return x
