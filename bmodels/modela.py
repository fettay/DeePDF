import torch.nn as nn
import math
from bmodels.layers import BBBConv1d, BBBLinearFactorial
import torch


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257

class BModelA(nn.Module):

    def __init__(self, input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH):
        super(BModelA, self).__init__()

        
        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)
        
        embedding_size = 16
        
        self.embed = nn.Embedding(input_height, embedding_size)         

        self.conv1 = BBBConv1d(self.q_logvar_init, self.p_logvar_init, embedding_size, 128, kernel_size=16, stride=4)
        self.soft1 = nn.Softplus()
        self.pooling = nn.MaxPool1d(int((input_length - 16)/4)+1)

        self.classifier = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 128, 1)

        layers = [self.conv1, self.soft1, self.pooling]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
            else:
                x = layer.forward(x)
        x = x.view(x.size(0), -1)
        x, _kl = self.classifier.fcprobforward(x)
        kl += _kl
        logits = x
        return logits, kl

    def __call__(self, x):
        return self.probforward(x)