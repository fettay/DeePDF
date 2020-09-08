import torch.nn as nn
import math
from bmodels.layers import ModuleWrapper, TransposeLayer, FlattenLayer, BBBConv1d, BBBLinear
import torch


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257

class BModelA(ModuleWrapper):

    def __init__(self, input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH):
        super(BModelA, self).__init__()

        
        embedding_size = 16
        
        self.embed = nn.Embedding(input_height, embedding_size)         
        self.transpose = TransposeLayer(1, 2)
        self.conv1 = BBBConv1d(embedding_size, 128, alpha_shape=(1,1), kernel_size=16, stride=16)
        self.relu = nn.Softplus()
        self.pooling = nn.MaxPool1d(int((input_length - 16)/16)+1)
        self.flatten = FlattenLayer(128)
        self.classifier = BBBLinear(128, 1)
