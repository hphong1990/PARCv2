import torch
import torch.nn as nn


class RecursiveFNO(nn.Module):
    def __init__(self, fno_model, steps):
        super(RecursiveFNO, self).__init__()
        self.fno_model = fno_model
        self.steps = steps
    
    def forward(self, x):
        output = []
        for i in range(self.steps):
            x = self.fno_model(x)
            output.append(x)  # b, c, y, x
        output = torch.cat(output, dim=1)
        return output