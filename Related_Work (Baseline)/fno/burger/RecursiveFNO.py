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
            vis = x[0, 2, 0, 0]
            x = self.fno_model(x)
            output.append(x[:, None, :, :, :])  # b, 1, c, y, x
            vis_x = torch.empty(x.shape[0], 1, x.shape[2], x.shape[3]).requires_grad_(False).cuda()
            vis_x[:, :, :, :] = vis
            x = torch.cat([x, vis_x], dim=1)
        output = torch.cat(output, dim=1)
        return output