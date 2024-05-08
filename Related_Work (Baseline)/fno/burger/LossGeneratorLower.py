import torch
import torch.nn as nn


# define the high-order finite difference kernels
lapl_op = [[[[   0, -1, 0],
             [  -1, 4,  -1],
             [   0, -1, 0]]]]

partial_y = [[[[   0, 0,   0],
               [ -1/2, 0, 1/2],
               [   0, 0,   0]]]]

partial_x = [[[[   0,-1/2,   0],
               [   0,   0,   0],
               [   0, 1/2,   0]]]]


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
        ''' Construct the derivatives, X = Width, Y = Height '''
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 3,
            name = 'laplace_operator').cuda()

        self.dx = Conv2dDerivative(
            DerFilter = partial_x,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dx_operator').cuda()

        self.dy = Conv2dDerivative(
            DerFilter = partial_y,
            resol = (dx*1),
            kernel_size = 3,
            name = 'dy_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 0, 1]]],
            resol = (dt*2),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output, R):
        # output: (b, s, c, y, x)
        # Modified to allow batched input
        # Pytorch convolution can only work on 3d/4d tensors, so some clever reshaping is necessary
        b, s, c, y, x = output.shape
        # spatial derivatives
        space_dev_input = output[:, 1:-1, :, :, :].reshape(-1, c, y, x)
        laplace_u = self.laplace(space_dev_input[:, 0:1, :, :]).reshape(b, s-2, 1, y-2, x-2)
        laplace_v = self.laplace(space_dev_input[:, 1:2, :, :]).reshape(b, s-2, 1, y-2, x-2)
        u_x = self.dx(space_dev_input[:, 0:1, :, :]).reshape(b, s-2, 1, y-2, x-2)
        u_y = self.dy(space_dev_input[:, 0:1, :, :]).reshape(b, s-2, 1, y-2, x-2)
        v_x = self.dx(space_dev_input[:, 1:2, :, :]).reshape(b, s-2, 1, y-2, x-2)
        v_y = self.dy(space_dev_input[:, 1:2, :, :]).reshape(b, s-2, 1, y-2, x-2)
        # temporal derivative
        time_dev_input = output[:, :, :, 1:-1, 1:-1]
        u, v = time_dev_input[:, :, 0:1, :, :], time_dev_input[:, :, 1:2, :, :]
        # ut
        u_conv1d = u.permute(0, 3, 4, 2, 1)   # (b, s, 1, y-2, x-2) --> (b, y-2, x-2, 1, s)
        u_conv1d = u_conv1d.reshape(-1, 1, s)
        u_t = self.dt(u_conv1d).reshape(b, y-2, x-2, 1, s-2)
        u_t = u_t.permute(0, 4, 3, 1, 2)      # (b, y-2, x-2, 1, s-2) --> (b, s-2, 1, y-2, x-2)
        # vt
        v_conv1d = v.permute(0, 3, 4, 2, 1)   # (b, s, 1, y-2, x-2) --> (b, y-2, x-2, 1, s)
        v_conv1d = v_conv1d.reshape(-1, 1, s)
        v_t = self.dt(v_conv1d).reshape(b, y-2, x-2, 1, s-2)
        v_t = v_t.permute(0, 4, 3, 1, 2)      # (b, y-2, x-2, 1, s-2) --> (b, s-2, 1, y-2, x-2)
        # u, v correct shape
        u = output[:, 1:-1, 0:1, 1:-1, 1:-1]  # (b, s-2, 1, y-2, x-2)
        v = output[:, 1:-1, 1:2, 1:-1, 1:-1]  # (b, s-2, 1, y-2, x-2)
        # 2D burgers eqn
        f_u = u_t + u * u_x + v * u_y - (1/R) * laplace_u
        f_v = v_t + u * v_x + v * v_y - (1/R) * laplace_v
        return f_u, f_v


def compute_loss(output, gt, loss_func, R, lam=0.9):
    ''' calculate the phycis loss '''
    mse_loss = nn.MSELoss(reduction='mean')
    # get physics loss
    output_padded = nn.functional.pad(output, (1, 1, 1, 1, 0, 0), 'constant', 0)
    f_u, f_v = loss_func.get_phy_Loss(output_padded, R)
    loss = (1.0 - lam) * mse_loss(f_u, torch.zeros_like(f_u).cuda()) + \
           (1.0 - lam) * mse_loss(f_v, torch.zeros_like(f_v).cuda()) + \
           lam * mse_loss(output[:, 1:-1, :, :, :], gt)
    return loss