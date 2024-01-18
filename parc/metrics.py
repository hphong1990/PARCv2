from scipy import stats as st
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import torch

# ================================================================= #
#                                                                   #
#               Metrics for Burgers problems                        #               
#                                                                   #
# ================================================================= #

class burgers_pde_loss:
    def __init__(self, dt=1.0, dx=1.0, **kwargs):
        super(burgers_pde_loss, self).__init__(**kwargs)
        self.dt = dt
        self.dx = dx
    
    def set_data(self, snapshot_data):
        self.snaphshot_data = snapshot_data
        
    def Laplacian(self, mat):
        dY, dX = torch.gradient(mat, spacing=self.dx)
        dYY, _ = torch.gradient(dY, spacing=self.dx)
        _, dXX = torch.gradient(dX, spacing=self.dx)
        laplacian = torch.add(dYY, dXX)
        return laplacian
    
    def TimeDerivative (self,U0, U1, U2):
        return ((U1 - U0) + (U2 - U1))/2.0/self.dt

    def SnapshotPdeLoss(self, U0, V0, U1, V1, U2, V2, nu=1.0):
        laplace_u = self.Laplacian(U1)
        laplace_v = self.Laplacian(V1)

        u_x, u_y = torch.gradient(U1, spacing=self.dx)
        v_x, v_y = torch.gradient(V1, spacing=self.dx)

        u_t_lhs = self.TimeDerivative(U0, U1, U2)
        v_t_lhs = self.TimeDerivative(V0, V1, V2)
        # governing equation
        u_t_rhs = nu * laplace_u - U1 * u_x - V1 * u_y
        v_t_rhs = nu * laplace_v - U1 * v_x - V1 * v_y

        delta_u = torch.abs(u_t_lhs - u_t_rhs)
        delta_v = torch.abs(v_t_lhs - v_t_rhs)
        return delta_u, delta_v
    
    def ComputePdeLoss(self, nu):
        sequence_length = len(self.snaphshot_data)
        fu = []
        fv = []
        for i in range(1, sequence_length-1):
            du, dv = self.SnapshotPdeLoss(self.snaphshot_data[i-1][0,:,:],
                                              self.snaphshot_data[i-1][1,:,:], 
                                              self.snaphshot_data[i][0,:,:],
                                              self.snaphshot_data[i][1,:,:], 
                                              self.snaphshot_data[i+1][0,:,:],
                                              self.snaphshot_data[i+1][1,:,:],nu)
            fu.append(du.reshape(1, du.shape[0], du.shape[1]))
            fv.append(dv.reshape(1, du.shape[0], du.shape[1]))
        fu = torch.cat(fu, dim=0)
        fv = torch.cat(fv, dim=0)
        return fu, fv

# ================================================================= #
#                                                                   #
#               Metrics for Navier-Stokes problems                  #               
#                                                                   #
# ================================================================= #

class ns_pde_loss:
    def __init__(self, dt=1.0, dx=1.0, **kwargs):
        super(ns_pde_loss, self).__init__(**kwargs)
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
    
    def ComputePdeLoss(self, rho, nu):
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
            fu.append(du.reshape(1, du.shape[0], du.shape[1]))
            fv.append(dv.reshape(1, du.shape[0], du.shape[1]))
            fp.append(dp.reshape(1, du.shape[0], du.shape[1]))
        fu = torch.cat(fu, dim=0)
        fv = torch.cat(fv, dim=0)
        fp = torch.cat(fp, dim=0)
        return fu, fv, fp

# ================================================================= #
#                                                                   #
#                   Metrics for EM problems                         #               
#                                                                   #
# ================================================================= #
class em_loss:
    def __init__(self, **kwargs):
        super(em_loss, self).__init__(**kwargs)

    def compute_KLD(y_true, y_pred):
        """compute KL-divergence
        :param y_true: (numpy)
        :param y_pred: (numpy)
        """

        mean_X = np.mean(y_true)
        sigma_X = np.std(y_true)

        mean_Y = np.mean(y_pred)
        sigma_Y = np.std(y_pred)

        v1 = sigma_X * sigma_X
        v2 = sigma_Y * sigma_Y
        a = np.log(sigma_Y / sigma_X)
        num = v1 + (mean_X - mean_Y) ** 2
        den = 2 * v2
        b = num / den
        return a + b - 0.5

    def compute_quantitative_evaluation_sensitivity(y_trues, y_preds):
        """Calculate average rmse, kld, and pearson correlation value
            across all time step of given sensitivity value derive from
            DNS (y_trues) w.r.t corresponding PARC predicted value (y_preds)

        :param y_trues: DNS ground truth sensitivity value
        :param y_preds: PARC predicted sensitivty value
        :return [0]:    (float) average rsme
        :return [1]:    (float) average kld
        :return [2]:    (float) average pearson correlation

        #"""

        pcc_list = []
        rmse_list = []
        kld_list = []
        ts = y_preds.shape[1]
        for i in range(ts):
            pcc = st.pearsonr(y_trues[:, i], y_preds[:, i])
            temp_rmse = sqrt(mean_squared_error(y_trues[:, i], y_preds[:, i]))
            kld = compute_KLD(y_trues[:, i], y_preds[:, i])
            pcc_list.append(pcc[0])
            rmse_list.append(temp_rmse)
            kld_list.append(kld)

        return np.mean(rmse_list), np.mean(kld_list), np.mean(pcc_list)


    def _calculate_hotspot_metric(Ts, n_timesteps=16):
        """calculates the hotspot temperature and area for single case
        :param test_data:   (numpy) temperature for single case with timesteps; [width, height, timesteps]
        :param n_timesteps: (int)   number of timesteps to calculate the sensitivity
        """
        hotspot_threshold = 850  # (K) Temperature threshold to distinguise between hotspot and non-hotspot area

        hotspot_areas = []
        hotspot_temperatures = []

        # Calculate area and avg hotspot temperature
        for i in range(n_timesteps):
            temp_i = Ts[:, :, i]
            hotspot_mask = temp_i > hotspot_threshold
            hotspot_area = np.count_nonzero(hotspot_mask)

            hotspot_area_rescaled = hotspot_area * ((2 * 1.5 / 256) ** 2)
            hotspot_areas.append(hotspot_area_rescaled)

            hotspot_temperature = temp_i * hotspot_mask

            if hotspot_area == 0:
                avg_hotspot_temperatures = 0.0
            else:
                avg_hotspot_temperatures = np.sum(hotspot_temperature) / hotspot_area
            hotspot_temperatures.append(avg_hotspot_temperatures)
        return hotspot_areas, hotspot_temperatures


    def calculate_hotspot_metric(T_cases, cases_range, n_timesteps):
        """calculates hotspot temperature and area for given cases
        :param T_cases:     (numpy) temperature fieds for different cases
        :param cases_range: (tuple) range of cases to test
        :param n_timesteps: (int) number of timesteps
        :return hs_temp:    (tuple) hotspot temperatures
                            [0] (float) mean
                            [1] (float) 5th percentile
                            [2] (float) 95th percentile
                            [3] (numpy) hotspot temperatures
        :return hs_area:    (tuple) hotspot area
                            [0] (float) mean
                            [1] (float) 5th percentile
                            [2] (float) 95th percentile
                            [3] (numpy) hotspot area
        """
        # calculate average hotspot area and temperature across cases
        hotspot_areas, hotspot_temperatures = [], []
        for i in range(cases_range[0], cases_range[1]):
            hotspot_areas_i, hotspot_temperatures_i = _calculate_hotspot_metric(
                T_cases[i, :, :, :], n_timesteps
            )
            hotspot_areas.append(hotspot_areas_i)
            hotspot_temperatures.append(hotspot_temperatures_i)

        hotspot_areas = np.array(hotspot_areas)
        hotspot_temperatures = np.array(hotspot_temperatures)

        mean_hotspot_temperatures = np.mean(hotspot_temperatures, axis=0)
        mean_hotspot_areas = np.mean(hotspot_areas, axis=0)

        temp_error1 = np.percentile(hotspot_temperatures, 95, axis=0)
        temp_error2 = np.percentile(hotspot_temperatures, 5, axis=0)
        area_error1 = np.percentile(hotspot_areas, 95, axis=0)
        area_error2 = np.percentile(hotspot_areas, 5, axis=0)

        hs_temp = (
            mean_hotspot_temperatures,
            temp_error1,
            temp_error2,
            hotspot_temperatures,
        )
        hs_area = (mean_hotspot_areas, area_error1, area_error2, hotspot_areas)
        return hs_temp, hs_area


    def calculate_hotspot_metric_rate_of_change(T_cases, cases_range, n_timesteps):
        """
        :param T_cases:         (numpy) temperature fields for different cases
        :param cases_range:     (tuple) range of cases to test
        :param n_timesteps:     (int)   number of timesteps
        :return rate_hs_temp:   (tuple) the rate of change for hotspot temperature
                                [0] (float) mean
                                [1] (float) 5th percentile
                                [2] (float) 95th percentile
                                [3] (numpy) rate of change for hotspot temperatures
        :return rate_hs_area:   (tuple) the rate of change for hotspot area
                                [0] (float) mean
                                [1] (float) 5th percentile
                                [2] (float) 95th percentile
                                [3] (numpy) rate of change for hotspot area
        """
        hotspot_areas, hotspot_temperatures = [], []
        for i in range(cases_range[0], cases_range[1]):
            hotspot_areas_i, hotspot_temperatures_i = _calculate_hotspot_metric(
                T_cases[i, :, :, :], n_timesteps
            )
            hotspot_areas.append(hotspot_areas_i)
            hotspot_temperatures.append(hotspot_temperatures_i)

        hotspot_areas = np.array(hotspot_areas)
        hotspot_temperatures = np.array(hotspot_temperatures)

        change_hotspot_areas = hotspot_areas[:, 1:] - hotspot_areas[:, 0:-1]
        change_hotspot_areas = change_hotspot_areas / (0.17)

        change_hotspot_temperatures = (
            hotspot_temperatures[:, 1:] - hotspot_temperatures[:, 0:-1]
        )
        change_hotspot_temperatures = change_hotspot_temperatures / (0.17)

        mean_Tdot_temperatures = np.mean(change_hotspot_temperatures, axis=0)
        mean_Tdot_areas = np.mean(change_hotspot_areas, axis=0)

        rate_temp_error1 = np.percentile(change_hotspot_temperatures, 95, axis=0)
        rate_temp_error2 = np.percentile(change_hotspot_temperatures, 5, axis=0)
        rate_area_error1 = np.percentile(change_hotspot_areas, 95, axis=0)
        rate_area_error2 = np.percentile(change_hotspot_areas, 5, axis=0)

        rate_hs_temp = (
            mean_Tdot_temperatures,
            rate_temp_error1,
            rate_temp_error2,
            change_hotspot_temperatures,
        )
        rate_hs_area = (
            mean_Tdot_areas,
            rate_area_error1,
            rate_area_error2,
            change_hotspot_areas,
        )
        return rate_hs_temp, rate_hs_area