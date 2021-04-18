import torch.nn as nn
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print("the current layer is :", x)
        print("the current size of layer is :", x.size())
        return x