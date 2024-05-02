import torch
import torch.nn as nn


class loss_generator:
    def __init__(self, dt=1.0, dx=1.0, **kwargs):
        super(loss_generator, self).__init__(**kwargs)
        self.dt = dt
        self.dx = dx
    
    def set_data(self, snapshot_data):
        self.snaphshot_data = snapshot_data
        
    def Laplacian(self, mat):
        dY, dX = torch.gradient(mat, spacing=self.dx)
        dYY, dYX = torch.gradient(dY, spacing=self.dx)
        dXY, dXX = torch.gradient(dX, spacing=self.dx)
        laplacian = torch.add(dYY, dXX)
        return laplacian
    
    def TimeDerivative (self,U0, U1, U2):
        return ((U1 - U0) + (U2 - U1))/2.0/self.dt

    def SnapshotPdeLoss(self, U0, V0, U1, V1, U2, V2, P1, rho=1.0, nu=1.0):
        laplace_u = self.Laplacian(U1)
        laplace_v = self.Laplacian(V1)

        u_x, u_y = torch.gradient(U1, spacing=self.dx)
        v_x, v_y = torch.gradient(V1, spacing=self.dx)
        p_x, p_y = torch.gradient(P1, spacing=self.dx)

        u_t_lhs = self.TimeDerivative(U0, U1, U2)
        v_t_lhs = self.TimeDerivative(V0, V1, V2)
        # governing equation
        u_t_rhs = nu * laplace_u - U1 * u_x - V1 * u_y - p_x
        v_t_rhs = nu * laplace_v - U1 * v_x - V1 * v_y - p_y
        p_t_rhs = u_x + v_y

        delta_u = torch.abs(u_t_lhs - u_t_rhs)
        delta_v = torch.abs(v_t_lhs - v_t_rhs)
        delta_p = torch.abs(p_t_rhs)
        return delta_u, delta_v, delta_p
    
    def ComputePdeLoss(self, rho, nu, mask):
        sequence_length = len(self.snaphshot_data)
        fu = []
        fv = []
        fp = []
        for i in range(1, sequence_length-1):
            du, dv, dp = self.SnapshotPdeLoss(self.snaphshot_data[i-1][0,:,:],
                                              self.snaphshot_data[i-1][1,:,:], 
                                              self.snaphshot_data[i][0,:,:],
                                              self.snaphshot_data[i][1,:,:], 
                                              self.snaphshot_data[i+1][0,:,:],
                                              self.snaphshot_data[i+1][1,:,:],
                                              self.snaphshot_data[i][2,:,:], rho, nu)
            du[mask] = 0.0
            dv[mask] = 0.0
            dp[mask] = 0.0
            fu.append(du.reshape(1, du.shape[0], du.shape[1]))
            fv.append(dv.reshape(1, du.shape[0], du.shape[1]))
            fp.append(dp.reshape(1, du.shape[0], du.shape[1]))
        fu = torch.cat(fu, dim=0)
        fv = torch.cat(fv, dim=0)
        fp = torch.cat(fp, dim=0)
        return fu, fv, fp


def compute_loss(output, loss_func, gt, rho, mask, nu=1.0, lam=0.1, fuv_lam=1e-6):
    mse_loss = nn.MSELoss(reduction='mean')
    # get physics loss
    fub = []
    fvb = []
    fpb = []
    for i in range(output.shape[0]):
        loss_func.set_data(output[i])
        f_u, f_v, f_p = loss_func.ComputePdeLoss(rho, nu, mask)
        fub.append(f_u.reshape(1, *f_u.shape))
        fvb.append(f_v.reshape(1, *f_u.shape))
        fpb.append(f_p.reshape(1, *f_u.shape))
    fub = torch.cat(fub, dim=0)
    fvb = torch.cat(fvb, dim=0)
    fpb = torch.cat(fpb, dim=0)
    mse_u = mse_loss(fub, torch.zeros_like(fub).cuda())
    mse_v = mse_loss(fvb, torch.zeros_like(fvb).cuda())
    mse_p = mse_loss(fpb, torch.zeros_like(fpb).cuda())
    mse_data = mse_loss(output[:, 1:-1, :, :, :], gt)
    # loss = lam*PDE + MSE
    loss = lam * mse_u * fuv_lam + lam * mse_v * fuv_lam + lam * mse_p + mse_data
    return loss, mse_u.detach().clone(), mse_v.detach().clone(), mse_p.detach().clone(), mse_data.detach().clone()
